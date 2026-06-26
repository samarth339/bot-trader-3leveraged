"""
Email sender for the trading bot.
Uses Gmail SMTP with an App Password.

Setup:
  1. Enable 2-Step Verification on your Google account.
  2. Go to https://myaccount.google.com/apppasswords
  3. Create an App Password (select "Mail" + "Other").
  4. Add to .env file:
       GMAIL_USER=samarth339@gmail.com
       GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

Usage:
  from send_email import send_email
  send_email("Subject", "Body text")
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

log = logging.getLogger("send_email")

# Load .env if python-dotenv is available, otherwise fall back to os.environ
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # Manual parse if python-dotenv not installed
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())


_load_env()

GMAIL_USER     = os.environ.get("GMAIL_USER", "samarth339@gmail.com").strip()
# Gmail shows app passwords as "xxxx xxxx xxxx xxxx" — accept that form by
# stripping the spaces a copy-paste leaves behind, plus any trailing newline a
# GitHub secret / .env line can carry. Both are common silent auth failures.
GMAIL_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "").strip().replace(" ", "")
SMTP_HOST      = "smtp.gmail.com"
SMTP_PORT      = 465   # SSL


def send_email(subject: str, body: str, to: str = None) -> bool:
    """
    Send a plain-text email via Gmail SMTP SSL.
    Returns True on success, False on failure.
    """
    if not GMAIL_PASSWORD:
        log.error(
            "GMAIL_APP_PASSWORD not set. "
            f"Add it to {Path(__file__).parent / '.env'} — see send_email.py for instructions."
        )
        return False

    recipient = to or GMAIL_USER

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = recipient
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, recipient, msg.as_string())
        log.info(f"Email sent: {subject}")
        return True
    except smtplib.SMTPAuthenticationError:
        # Show enough to diagnose WITHOUT leaking the secret: the account in use
        # and the length of the credential (a valid Gmail app password is 16
        # chars after spaces are stripped). Wrong length ⇒ secret is malformed;
        # right length but still failing ⇒ revoked/expired ⇒ regenerate it.
        log.error(
            "Gmail authentication failed (SMTPAuthenticationError). "
            f"Account='{GMAIL_USER}', app-password length={len(GMAIL_PASSWORD)} "
            "(expected 16). Must be a Gmail App Password, not the account password. "
            "Local: set GMAIL_USER / GMAIL_APP_PASSWORD in .env. "
            "CI: update the GMAIL_USER / GMAIL_APP_PASSWORD GitHub Actions secrets. "
            "If length is 16 and it still fails, the password was revoked — "
            "regenerate at https://myaccount.google.com/apppasswords"
        )
        return False
    except Exception as e:
        log.error(f"Email send failed: {e}")
        return False


def send_pending_emails():
    """
    Send any queued emails from pending_email.json / pending_weekly_email.json.
    Called after shadow_mode.py writes those files.
    """
    import json
    from pathlib import Path

    logs_dir = Path(__file__).parent / "logs"
    for fname in ("pending_email.json", "pending_weekly_email.json"):
        pending = logs_dir / fname
        if not pending.exists():
            continue
        try:
            data = json.loads(pending.read_text())
            ok = send_email(data["subject"], data["body"], data.get("to"))
            if ok:
                pending.unlink()
        except Exception as e:
            log.error(f"Failed to process {fname}: {e}")
