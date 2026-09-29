# Phase 5 VPS Runbook — IBKR paper execution

Reliable ET scheduling + headless Gateway. Replaces GitHub Actions for execution
(GitHub `schedule:` is best-effort and dispatches hours late — unusable for MOC).

## What you need
- A small always-on Linux VPS (2 GB+ RAM; ~$5/mo Hetzner is plenty). Docker installed.
- Your **paper** IBKR user id + password (separate paper login).
- The repo at `/opt/bot-trader-3leveraged`, Python deps installed, `.env` with GMAIL_* (chmod 600).

## 1. Headless Gateway (ibeam)
```bash
export IBEAM_ACCOUNT=<paper-user-id>          # host env only — never committed
export IBEAM_PASSWORD=<paper-password>
cd /opt/bot-trader-3leveraged/deploy/vps
docker compose up -d
docker compose logs -f ibeam                  # confirm "Gateway running", API :4002
```
Confirm the paper login has **no 2FA** (headless can't answer a push). ibeam auto-handles
IBKR's daily re-login. Port 4002 is bound to 127.0.0.1 — never expose it publicly.

## 2. Schedule (systemd, DST-safe)
```bash
sudo cp ibkr-executor.{service,timer} watchdog.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ibkr-executor.timer watchdog.timer
systemctl list-timers | grep -E 'ibkr|watchdog'   # verify next fire times in ET
```
Executor fires Mon–Fri 15:45 ET (MOC window); watchdog 16:30 ET (strict same-day alert).

## 3. Staged rollout (do NOT skip)
1. **Fill test** — once, manually, in the window: `python3 -m ibkr.executor --paper --dry-run`
   then a real 1-share fill. Confirm submit → fill → post-fill reconciliation (A5) is clean.
2. **Full daily on paper** — let the timer run it for weeks. Watch: fills vs sim, slippage,
   reconciliation alerts, and behaviour **through at least one real volatility event**.
3. **Real money** — only after all of the above: small size (10%), kill switch armed,
   and a separate explicit authorization. Storing trade-capable creds on a VPS is a real
   attack surface at that point — harden the box (firewall, no public ports, key-only SSH).

## Kill switch
`touch logs/ibkr_kill.flag` (or via the Admin panel) halts the executor and flattens on the
next run. Remove the file to resume.
