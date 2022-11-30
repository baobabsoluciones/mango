import os
import smtplib
import ssl
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from os.path import basename


class EmailSender:
    def __init__(
        self, host: str = None, port: int = None, user: str = None, password: str = None
    ):
        self.host = host if host is not None else os.getenv("EMAIL_HOST")
        self.port = port if port is not None else os.getenv("EMAIL_PORT")
        self.user = user if user is not None else os.getenv("EMAIL_USER")
        self.password = (
            password if password is not None else os.getenv("EMAIL_PASSWORD")
        )

    def send_email_with_attachments(
        self, destination: str, subject: str, attachments: list = None
    ):

        msg = MIMEMultipart()
        msg["From"] = self.user
        msg["To"] = destination
        msg["Date"] = formatdate(localtime=True)
        msg["Subject"] = subject

        msg.attach(MIMEText(""))

        for at in attachments:
            with open(at, "rb") as fp:
                part = MIMEApplication(fp.read(), Name=basename(at))

            part["Content-Disposition"] = f"attachment; filename={basename(at)}"
            msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.host, port=465, context=context) as server:
            server.login(self.user, self.password)
            server.send_message(msg, self.user, destination)
