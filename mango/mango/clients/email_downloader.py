import email
import imaplib
import logging as log
import os


class EmailDownloader:
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
        Method to close the connection to the email provider
        """
        self.connection.close()

    def fetch_unread_emails(self) -> list:
        """
        This method gets all the unread emails, downloads them and marks them as read.

        :return: a list with the emails
        :rtype: list
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
        Method that given an emails downloads the attachments found on them to the local disk

        :param msg: an email
        :param str download_folder: the folder where the attachments have to be downloaded
        :return: a list of the paths of the downloaded attachments
        :rtype: list
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
