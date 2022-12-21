import os
import smtplib
import ssl
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from os.path import basename


class EmailSender:
    """
    This class can be used to send emails to a given destination address from a given address.
    """

    def __init__(
        self, host: str = None, port: int = None, user: str = None, password: str = None
    ):
        self.host = host if host is not None else os.getenv("EMAIL_HOST", None)
        self.port = port if port is not None else os.getenv("EMAIL_PORT", None)
        self.user = user if user is not None else os.getenv("EMAIL_USER", None)
        self.password = (
            password if password is not None else os.getenv("EMAIL_PASSWORD", None)
        )

        if (
            self.host is None
            or self.port is None
            or self.user is None
            or self.password is None
        ):
            raise ValueError(
                "Some configuration (host, port, user or password) is missing."
            )

    def send_email_with_attachments(
        self, destination: str, subject: str, body: str = None, attachments: list = None
    ):
        """
        Method to email a destination address with a subject, a body and attachments

        :param str destination: the email address that receives the email
        :param str subject: the subject of the email
        :param str body: the body of the email
        :param list attachments: the list with the paths of the attachments for the email
        """

        msg = MIMEMultipart()
        msg["From"] = self.user
        msg["To"] = destination
        msg["Date"] = formatdate(localtime=True)
        msg["Subject"] = subject

        msg.attach(MIMEText(f"{body}"))

        for at in attachments:
            with open(at, "rb") as fp:
                part = MIMEApplication(fp.read(), Name=basename(at))

            part["Content-Disposition"] = f"attachment; filename={basename(at)}"
            msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.host, port=465, context=context) as server:
            server.login(self.user, self.password)
            server.send_message(msg, self.user, destination)
