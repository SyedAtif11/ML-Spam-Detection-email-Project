# step7_email_bot.py
# Assignment 4 — Live Email Classification Bot
# -------------------------------------------------------
# This bot:
#   1. Connects to Gmail via IMAP (read mail)
#   2. Finds UNSEEN messages
#   3. Cleans + vectorizes the body with the same pipeline
#      used during training
#   4. Predicts label + confidence
#   5. Sends an auto-reply via SMTP with the result
#   6. Marks the original message as SEEN
#   7. Polls again every 30 seconds
#
# Setup (do ONCE before running):
#   • Enable "Less Secure Apps" OR create a Gmail App Password
#     (Google Account → Security → 2-Step Verification → App Passwords)
#   • Set two environment variables:
#       Windows:  set BOT_EMAIL=yourbot@gmail.com
#                 set BOT_APP_PASSWORD=xxxx xxxx xxxx xxxx
#       Mac/Linux: export BOT_EMAIL=...
#                  export BOT_APP_PASSWORD=...
# -------------------------------------------------------

import imaplib
import smtplib
import email
import os
import re
import string
import time
import joblib
import nltk

from email.mime.text    import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header       import decode_header
from nltk.corpus        import stopwords
from nltk.stem          import PorterStemmer

# ── Download NLTK data if needed ─────────────────────────
nltk.download("stopwords", quiet=True)

# ── Credentials from environment ─────────────────────────
BOT_EMAIL    = os.environ.get("BOT_EMAIL",        "your_bot@gmail.com")
BOT_PASSWORD = os.environ.get("BOT_APP_PASSWORD", "your_app_password")

IMAP_HOST = "imap.gmail.com"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# ── Load model + vectorizer ───────────────────────────────
print("[init] Loading model and vectorizer …")
model      = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
print("[init] Ready!\n")

# ── Text cleaning (must match step2_clean.py) ────────────
STOP    = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens
              if w not in STOP and len(w) > 2]
    return " ".join(tokens)


# ── Helper: decode email subject ─────────────────────────
def decode_subject(raw_subject: str) -> str:
    parts = decode_header(raw_subject)
    subject = ""
    for part, enc in parts:
        if isinstance(part, bytes):
            subject += part.decode(enc or "utf-8", errors="ignore")
        else:
            subject += part
    return subject


# ── Helper: extract plain-text body ──────────────────────
def get_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="ignore")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode("utf-8", errors="ignore")
    return ""


# ── Core: classify + reply ────────────────────────────────
def process_message(mail, msg_id, msg):
    sender  = msg["From"]
    subject = decode_subject(msg.get("Subject", "(no subject)"))
    body    = get_body(msg)

    if not body.strip():
        print(f"  [skip] empty body from {sender}")
        return

    # Classify
    cleaned = clean_text(body)
    vec     = vectorizer.transform([cleaned])
    label   = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    conf    = max(proba) * 100

    print(f"  [classify] {sender}  →  {label.upper()}  ({conf:.1f}% confidence)")

    # Compose reply
    reply_body = (
        f"Hello,\n\n"
        f"Your message has been automatically classified:\n\n"
        f"  Result     : {label.upper()}\n"
        f"  Confidence : {conf:.1f}%\n\n"
        f"Original subject: {subject}\n\n"
        f"— Auto-Classifier Bot (Assignment 4)\n"
    )

    reply_msg = MIMEMultipart()
    reply_msg["From"]    = BOT_EMAIL
    reply_msg["To"]      = sender
    reply_msg["Subject"] = f"Re: {subject} — classified as {label.upper()}"
    reply_msg.attach(MIMEText(reply_body, "plain"))

    # Send via SMTP
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.starttls()
        smtp.login(BOT_EMAIL, BOT_PASSWORD)
        smtp.sendmail(BOT_EMAIL, sender, reply_msg.as_string())

    # Mark as read so we don't re-process
    mail.store(msg_id, "+FLAGS", "\\Seen")
    print(f"  [replied]  reply sent to {sender}")


# ── Main polling loop ─────────────────────────────────────
def check_inbox():
    mail = imaplib.IMAP4_SSL(IMAP_HOST)
    mail.login(BOT_EMAIL, BOT_PASSWORD)
    mail.select("inbox")

    status, messages = mail.search(None, "UNSEEN")
    if status != "OK" or not messages[0]:
        print("[poll] no new messages.")
        mail.logout()
        return

    msg_ids = messages[0].split()
    print(f"[poll] {len(msg_ids)} new message(s).")

    for msg_id in msg_ids:
        _, data = mail.fetch(msg_id, "(RFC822)")
        msg = email.message_from_bytes(data[0][1])
        try:
            process_message(mail, msg_id, msg)
        except Exception as err:
            print(f"  [error] {err}")

    mail.logout()


if __name__ == "__main__":
    print("Bot is running. Press Ctrl+C to stop.\n")
    while True:
        try:
            check_inbox()
        except Exception as e:
            print("Connection error:", e)
        time.sleep(30)   # poll every 30 seconds
