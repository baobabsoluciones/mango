import email
import imaplib
import os

from mango.logging import get_configured_logger

log = get_configured_logger(__name__)


class EmailDownloader:
    """
    Client for downloading emails and attachments from IMAP servers.

    This class provides functionality to connect to IMAP email servers,
    download unread emails, and save attachments to the local filesystem.
    Supports SSL connections and automatic email marking as read.

    :param host: IMAP server hostname (default: from EMAIL_HOST environment variable)
    :type host: str, optional
    :param user: Email username (default: from EMAIL_USER environment variable)
    :type user: str, optional
    :param password: Email password (default: from EMAIL_PASSWORD environment variable)
    :type password: str, optional
    :raises ValueError: If any required configuration is missing

    Example:
        >>> downloader = EmailDownloader(
        ...     host="imap.gmail.com",
        ...     user="user@gmail.com",
        ...     password="password"
        ... )
        >>> emails = downloader.fetch_unread_emails()
    """

    def __init__(self, host: str = None, user: str = None, password: str = None):
        self.host = host if host is not None else os.getenv("EMAIL_HOST", None)
        self.user = user if user is not None else os.getenv("EMAIL_USER", None)
        self.password = (
            password if password is not None else os.getenv("EMAIL_PASSWORD", None)
        )

        if self.host is None or self.user is None or self.password is None:
            raise ValueError("Some configuration (host, user or password) is missing.")

        self.connection = imaplib.IMAP4_SSL(self.host)
        self.connection.login(self.user, self.password)
        self.connection.select(readonly=False)

    def close_connection(self):
        """
        Close the connection to the IMAP email server.

        Properly closes the IMAP connection to free up resources.
        Should be called when finished with email operations.

        Example:
            >>> downloader = EmailDownloader()
            >>> # ... use downloader ...
            >>> downloader.close_connection()
        """
        self.connection.close()

    def fetch_unread_emails(self) -> list:
        """
        Fetch all unread emails from the IMAP server.

        Downloads all unread emails from the server and automatically marks
        them as read. Returns a list of email message objects that can be
        processed further.

        :return: List of email message objects
        :rtype: list
        :raises Exception: If IMAP operations fail

        Example:
            >>> emails = downloader.fetch_unread_emails()
            >>> for email in emails:
            ...     print(f"Subject: {email.get('Subject')}")
        """
        emails = []
        result, messages = self.connection.search(None, "UNSEEN")
        if result == "OK":
            for message in messages[0].decode().split(" "):
                try:
                    ret, data = self.connection.fetch(message, "(RFC822)")
                except:
                    log.info("No new emails to read")
                    self.close_connection()
                    return emails

                msg = email.message_from_bytes(data[0][1])
                if not isinstance(msg, str):
                    emails.append(msg)
                response, data = self.connection.store(message, "+FLAGS", "\\Seen")
            self.close_connection()
            return emails
        log.error("Failed to retrieve emails")
        self.close_connection()
        return emails

    @staticmethod
    def save_attachments(msg, download_folder: str = ".") -> list:
        """
        Save email attachments to the local filesystem.

        Extracts and saves all attachments from an email message to the
        specified local directory. Skips files that already exist to
        avoid overwriting.

        :param msg: Email message object containing attachments
        :type msg: email.message.Message
        :param download_folder: Local directory path to save attachments
        :type download_folder: str
        :return: List of file paths where attachments were saved
        :rtype: list

        Example:
            >>> emails = downloader.fetch_unread_emails()
            >>> for email in emails:
            ...     attachments = EmailDownloader.save_attachments(email, "./downloads")
            ...     print(f"Saved {len(attachments)} attachments")
        """
        att_path = []
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get("Content-Disposition") is None:
                continue

            file_name = part.get_filename()

            path = f"{download_folder}/{file_name}"
            if not os.path.isfile(path):
                fp = open(path, "wb")
                fp.write(part.get_payload(decode=True))
                fp.close()

            att_path.append(path)

        return att_path
