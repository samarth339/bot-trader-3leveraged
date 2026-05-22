"""
client.py — IB Gateway Connection Layer
========================================
Wraps ib_insync IB() with:
  - Exponential backoff reconnect (0 → 10 → 30 → 60 → 120s)
  - Background heartbeat (60s interval, 5s timeout)
  - Context manager interface
  - Full event logging (connect / disconnect / error)

Ports:
  IB Gateway LIVE  : 4001   (clientId=10 reserved for bot)
  IB Gateway PAPER : 4002   (clientId=11 reserved for bot)
  TWS LIVE         : 7496
  TWS PAPER        : 7497
"""

import time
import logging
import threading
from dataclasses import dataclass
from typing import Optional

from ib_insync import IB, util

logger = logging.getLogger("ibkr.client")

# ── Port constants ─────────────────────────────────────────────────────────────
GATEWAY_LIVE_PORT  = 4001
GATEWAY_PAPER_PORT = 4002
TWS_LIVE_PORT      = 7496
TWS_PAPER_PORT     = 7497

# ── Reconnect schedule ─────────────────────────────────────────────────────────
RECONNECT_DELAYS = [0, 10, 30, 60, 120]   # seconds before each attempt

# ── Heartbeat ─────────────────────────────────────────────────────────────────
HEARTBEAT_INTERVAL = 60    # seconds between checks
HEARTBEAT_TIMEOUT  = 5     # seconds to wait for server response


@dataclass
class ConnectionConfig:
    host:      str   = "127.0.0.1"
    port:      int   = GATEWAY_PAPER_PORT
    client_id: int   = 11           # 10=live bot, 11=paper bot
    paper:     bool  = True
    timeout:   float = 20.0         # seconds for connect()


class IBClient:
    """
    Managed IB Gateway connection with auto-reconnect and heartbeat.

    Usage (context manager — preferred):
        config = ConnectionConfig(port=4002, paper=True)
        with IBClient(config) as client:
            ib = client.ib        # ib_insync IB instance, ready to use

    Usage (manual):
        client = IBClient(config)
        client.connect(deadline_secs=300)
        ...
        client.disconnect()
    """

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.ib     = IB()
        self._connected          = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat     = threading.Event()

        # Wire IBKR event callbacks
        self.ib.connectedEvent    += self._on_connect
        self.ib.disconnectedEvent += self._on_disconnect
        self.ib.errorEvent        += self._on_error

    # ── Context manager ────────────────────────────────────────────────────────
    def __enter__(self) -> "IBClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.disconnect()
        return False   # do not suppress exceptions

    # ── Connection ─────────────────────────────────────────────────────────────
    def connect(self, deadline_secs: Optional[int] = None) -> bool:
        """
        Connect to IB Gateway with exponential backoff.

        Args:
            deadline_secs: hard deadline — give up if not connected in time.
                           None = try all RECONNECT_DELAYS without time limit.

        Returns:
            True if connected, False if all attempts failed.
        """
        start = time.monotonic()

        for attempt, delay in enumerate(RECONNECT_DELAYS):
            if delay > 0:
                elapsed = time.monotonic() - start
                if deadline_secs is not None and (elapsed + delay) > deadline_secs:
                    logger.error(
                        f"Connection deadline ({deadline_secs}s) would be exceeded "
                        f"— aborting after {attempt} attempts"
                    )
                    return False
                logger.info(
                    f"Reconnect backoff: waiting {delay}s "
                    f"(attempt {attempt + 1}/{len(RECONNECT_DELAYS)})"
                )
                time.sleep(delay)

            try:
                logger.info(
                    f"Connecting → {self.config.host}:{self.config.port}  "
                    f"clientId={self.config.client_id}  "
                    f"{'PAPER' if self.config.paper else 'LIVE'}"
                )
                # ib_insync needs an event loop; startLoop() is safe to call
                # multiple times — it's a no-op if already running
                util.startLoop()
                self.ib.connect(
                    host=self.config.host,
                    port=self.config.port,
                    clientId=self.config.client_id,
                    timeout=self.config.timeout,
                    readonly=False,
                )
                self._connected = True
                self._start_heartbeat()
                logger.info(
                    f"Connected ✓  serverVersion={self.ib.serverVersion()}  "
                    f"account={self.ib.wrapper.accounts}"
                )
                return True

            except Exception as exc:
                logger.warning(
                    f"Connection attempt {attempt + 1} failed: {exc}"
                )
                self._connected = False

        logger.error("All reconnection attempts exhausted — giving up")
        return False

    def disconnect(self):
        """Cleanly stop heartbeat and disconnect from IB Gateway."""
        self._stop_heartbeat.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)
        if self.ib.isConnected():
            self.ib.disconnect()
        self._connected = False
        logger.info("Disconnected from IB Gateway")

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    # ── Heartbeat ──────────────────────────────────────────────────────────────
    def _start_heartbeat(self):
        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="ibkr-heartbeat"
        )
        self._heartbeat_thread.start()
        logger.debug("Heartbeat thread started")

    def _heartbeat_loop(self):
        """Background thread: ping server every HEARTBEAT_INTERVAL seconds."""
        while not self._stop_heartbeat.wait(HEARTBEAT_INTERVAL):
            # Stop flag set by disconnect() — exit immediately without reconnecting
            if self._stop_heartbeat.is_set():
                return
            if not self.ib.isConnected():
                logger.warning("Heartbeat: connection lost — triggering reconnect")
                self.connect(deadline_secs=300)
                return   # let the new connect() start a fresh heartbeat

            try:
                server_time = self.ib.reqCurrentTime()
                logger.debug(f"Heartbeat ✓  server_time={server_time}")
            except Exception as exc:
                logger.warning(f"Heartbeat reqCurrentTime failed: {exc}")

    # ── Event callbacks ────────────────────────────────────────────────────────
    def _on_connect(self):
        logger.info("IB event: connectedEvent")
        self._connected = True

    def _on_disconnect(self):
        logger.warning("IB event: disconnectedEvent — connection dropped")
        self._connected = False

    def _on_error(self, req_id: int, error_code: int, error_string: str, contract):
        # IBKR sends informational messages as "errors" — filter them
        INFO_CODES = {
            2104,   # Market data farm connection OK
            2106,   # HMDS data farm connection OK
            2108,   # Market data farm connection inactive
            2119,   # Market data farm is connecting
            2158,   # Sec-def data farm connection OK
            2100,   # Account information update
        }
        if error_code in INFO_CODES:
            logger.debug(f"IB info [{error_code}]: {error_string}")
        elif error_code == 1100:
            logger.error("IB [1100]: Connectivity to IB lost — reconnect pending")
        elif error_code == 1102:
            logger.info("IB [1102]: Connectivity to IB restored")
        elif error_code == 502:
            logger.error(
                f"IB [502]: Cannot connect to IB Gateway at "
                f"{self.config.host}:{self.config.port} — is Gateway running?"
            )
        else:
            logger.error(
                f"IB error [{error_code}] reqId={req_id}: {error_string}  "
                f"contract={contract}"
            )
