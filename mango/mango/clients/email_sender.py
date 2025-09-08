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
    Client for sending emails with attachments via SMTP.

    This class provides functionality to send emails through SMTP servers
    with support for attachments, HTML/text content, and SSL encryption.
    Configuration can be provided directly or through environment variables.

    :param host: SMTP server hostname (default: from EMAIL_HOST environment variable)
    :type host: str, optional
    :param port: SMTP server port (default: from EMAIL_PORT environment variable)
    :type port: int, optional
    :param user: Email username (default: from EMAIL_USER environment variable)
    :type user: str, optional
    :param password: Email password (default: from EMAIL_PASSWORD environment variable)
    :type password: str, optional
    :raises ValueError: If any required configuration is missing

    Example:
        >>> sender = EmailSender(
        ...     host="smtp.gmail.com",
        ...     port=465,
        ...     user="sender@gmail.com",
        ...     password="password"
        ... )
        >>> sender.send_email_with_attachments(
        ...     destination="recipient@example.com",
        ...     subject="Test Email",
        ...     body="Hello World",
        ...     attachments=["file1.pdf", "file2.txt"]
        ... )
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
        Send an email with optional attachments via SMTP.

        Creates and sends an email message with the specified content and attachments.
        Uses SSL encryption for secure transmission. Attachments are read from
        local file paths and attached to the email.

        :param destination: Email address of the recipient
        :type destination: str
        :param subject: Subject line of the email
        :type subject: str
        :param body: Email body content (plain text)
        :type body: str, optional
        :param attachments: List of file paths to attach to the email
        :type attachments: list, optional
        :raises FileNotFoundError: If any attachment file is not found
        :raises Exception: If SMTP connection or sending fails

        Example:
            >>> sender.send_email_with_attachments(
            ...     destination="user@example.com",
            ...     subject="Report",
            ...     body="Please find the report attached.",
            ...     attachments=["report.pdf", "data.xlsx"]
            ... )
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
