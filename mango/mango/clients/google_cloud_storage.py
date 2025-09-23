import os

from google.api_core.exceptions import NotFound
from google.cloud import storage
from mango.clients.cloud_storage import CloudStorage
from mango.logging import get_configured_logger

log = get_configured_logger(__name__)


class GoogleCloudStorage(CloudStorage):
    """
    Google Cloud Storage client for managing files and objects in GCS buckets.

    This class provides a comprehensive interface for Google Cloud Storage operations
    including uploading, downloading, deleting, copying, and moving files. It extends
    the abstract CloudStorage class with GCS-specific functionality.

    :param secrets_file: Path to Google Cloud service account JSON file
    :type secrets_file: str
    :param bucket_name: Name of the GCS bucket to work with
    :type bucket_name: str
    :raises ValueError: If secrets file is missing

    Example:
        >>> gcs = GoogleCloudStorage(
        ...     secrets_file="/path/to/service-account.json",
        ...     bucket_name="my-bucket"
        ... )
        >>> gcs.upload_from_filename("local_file.txt", "remote_file.txt")
    """

    def __init__(self, secrets_file, bucket_name):
        secrets = (
            secrets_file
            if secrets_file is not None
            else os.getenv("SECRETS_FILE", None)
        )

        if secrets is None:
            raise ValueError("The secrets file is missing.")
        self.connection = storage.Client.from_service_account_json(secrets)

        self.bucket_name = (
            bucket_name if bucket_name is not None else os.getenv("BUCKET_NAME")
        )

        self.bucket = self.connection.bucket(self.bucket_name)

    def upload_object(self, contents, blob_name: str):
        """
        Upload Python object contents to Google Cloud Storage.

        Uploads raw data/contents directly to the specified blob in the GCS bucket.
        The contents must be convertible to string or bytes format.

        :param contents: The data content to upload (string, bytes, or serializable object)
        :type contents: Union[str, bytes, object]
        :param blob_name: Name/path of the blob in the GCS bucket
        :type blob_name: str

        Example:
            >>> gcs.upload_object("Hello World", "greeting.txt")
            >>> gcs.upload_object(b"Binary data", "data.bin")
        """
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(contents)
        log.info(f"Uploaded contents to blob {blob_name}")

    def upload_from_filename(self, file_name: str, blob_name: str):
        """
        Upload file from local filesystem to Google Cloud Storage.

        Reads a file from the local filesystem and uploads its contents
        to the specified blob in the GCS bucket.

        :param file_name: Path to the local file to upload
        :type file_name: str
        :param blob_name: Name/path of the blob in the GCS bucket
        :type blob_name: str
        :raises FileNotFoundError: If the local file does not exist

        Example:
            >>> gcs.upload_from_filename("/path/to/local/file.txt", "remote/file.txt")
        """

        if not os.path.exists(file_name):
            log.error(f"The file {file_name} does not exist")
            raise FileNotFoundError(f"The file {file_name} does not exist")

        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(file_name)
        log.info(f"Uploaded file {file_name} to blob {blob_name}")

    def upload_from_file(self, file, blob_name: str):
        """
        Upload file from an open file handle to Google Cloud Storage.

        Uploads data from an already open file handle to the specified blob
        in the GCS bucket. The file handle must be in binary read mode.

        :param file: Open file handle in binary read mode
        :type file: file-like object
        :param blob_name: Name/path of the blob in the GCS bucket
        :type blob_name: str

        Example:
            >>> with open("data.txt", "rb") as f:
            ...     gcs.upload_from_file(f, "remote/data.txt")
        """
        blob = self.bucket.blob(blob_name)
        blob.upload_from_file(file)
        log.info(f"Uploaded file to blob {blob_name}")

    def rename_file(self, blob_name: str, new_name: str):
        """
        Rename an existing blob in Google Cloud Storage.

        Changes the name of an existing blob within the same bucket.
        This operation creates a new blob with the new name and deletes the old one.

        :param blob_name: Current name of the blob to rename
        :type blob_name: str
        :param new_name: New name for the blob
        :type new_name: str
        :raises NotFound: If the source blob does not exist

        Example:
            >>> gcs.rename_file("old_name.txt", "new_name.txt")
        """

        blob = self.bucket.blob(blob_name)
        try:
            new_blob = self.bucket.rename_blob(blob, new_name)
            log.info(f"Blob {blob.name} has been renamed to {new_blob.name}")
        except NotFound as e:
            log.error(f"Blob {blob_name} does not exist")
            raise e

    def download_to_object(self, blob_name: str):
        """
        Download blob contents as a Python object from Google Cloud Storage.

        Downloads the blob contents and returns them as bytes. The contents
        can be converted to string or other formats as needed.

        :param blob_name: Name of the blob to download
        :type blob_name: str
        :return: Blob contents as bytes
        :rtype: bytes
        :raises NotFound: If the blob does not exist

        Example:
            >>> content = gcs.download_to_object("file.txt")
            >>> text = content.decode('utf-8')
        """
        blob = self.bucket.blob(blob_name)
        try:
            contents = blob.download_as_string()
        except NotFound as e:
            log.error(f"Blob {blob_name} does not exist")
            raise e

        log.info(f"Blob {blob_name} has been downloaded")
        return contents

    def download_to_file(self, blob_name: str, destination_path: str):
        """
        Download blob from Google Cloud Storage to local filesystem.

        Downloads the specified blob and saves it to the local filesystem
        at the given destination path.

        :param blob_name: Name of the blob to download
        :type blob_name: str
        :param destination_path: Local file path where the blob will be saved
        :type destination_path: str
        :raises NotFound: If the blob does not exist

        Example:
            >>> gcs.download_to_file("remote/file.txt", "/local/path/file.txt")
        """
        blob = self.bucket.blob(blob_name)
        try:
            blob.download_to_filename(destination_path)
        except NotFound as e:
            log.error(f"Blob {blob_name} does not exist")
            raise e
        log.info(f"Blob {blob_name} has been downloaded to file {destination_path}")

    def delete_file(self, blob_name: str):
        """
        Delete a blob from Google Cloud Storage bucket.

        Permanently removes the specified blob from the GCS bucket.
        This operation cannot be undone.

        :param blob_name: Name of the blob to delete
        :type blob_name: str
        :raises NotFound: If the blob does not exist

        Example:
            >>> gcs.delete_file("unwanted_file.txt")
        """
        blob = self.bucket.blob(blob_name)
        try:
            blob.delete()
        except NotFound as e:
            log.error(f"Blob {blob_name} does not exist")
            raise e
        log.info(f"The blob {blob_name} in bucket {self.bucket_name} has been deleted")

    def list_files(self):
        """
        List all files in the Google Cloud Storage bucket.

        Retrieves a list of all blob names currently stored in the bucket.
        This includes all files regardless of their location within the bucket.

        :return: List of blob names in the bucket
        :rtype: list

        Example:
            >>> files = gcs.list_files()
            >>> print(f"Bucket contains {len(files)} files")
        """
        blobs = self.connection.list_blobs(self.bucket_name)
        return [blob.name for blob in blobs]

    def copy_file(
        self,
        blob_name: str,
        destination_bucket: str = None,
        destination_blob_name: str = None,
    ):
        """
        Copy a blob to another bucket in Google Cloud Storage.

        Creates a copy of the specified blob in the destination bucket.
        If no destination bucket is specified, uses the environment variable
        DESTINATION_BUCKET_NAME. If no new name is provided, keeps the original name.

        :param blob_name: Name of the source blob to copy
        :type blob_name: str
        :param destination_bucket: Name of the destination bucket
        :type destination_bucket: str, optional
        :param destination_blob_name: New name for the blob in destination bucket
        :type destination_blob_name: str, optional
        :raises NotFound: If the source blob does not exist

        Example:
            >>> gcs.copy_file("source.txt", "other-bucket", "copied.txt")
        """
        if destination_bucket is not None:
            destination_bucket = self.connection.bucket(destination_bucket)
        else:
            destination_bucket = self.connection.bucket(
                os.getenv("DESTINATION_BUCKET_NAME")
            )

        if destination_blob_name is None:
            destination_blob_name = blob_name

        blob = self.bucket.blob(blob_name)
        try:
            blob_copy = self.bucket.copy_blob(
                blob, destination_bucket, destination_blob_name
            )
            log.info(
                f"Blob {blob_name} has been moved from bucket {self.bucket_name} "
                f"to bucket {destination_bucket} as blob {blob_copy.name}"
            )
        except NotFound as e:
            log.error(f"Blob {blob_name} does not exist")
            raise e

    def move_file(
        self,
        blob_name: str,
        destination_bucket: str = None,
        destination_blob_name: str = None,
    ):
        """
        Move a blob from one bucket to another in Google Cloud Storage.

        Moves the specified blob to the destination bucket by copying it
        and then deleting the original. If no destination bucket is specified,
        uses the environment variable DESTINATION_BUCKET_NAME.

        :param blob_name: Name of the source blob to move
        :type blob_name: str
        :param destination_bucket: Name of the destination bucket
        :type destination_bucket: str, optional
        :param destination_blob_name: New name for the blob in destination bucket
        :type destination_blob_name: str, optional
        :raises NotFound: If the source blob does not exist

        Example:
            >>> gcs.move_file("source.txt", "archive-bucket", "moved.txt")
        """
        if destination_bucket is not None:
            destination_bucket = self.connection.bucket(destination_bucket)
        else:
            destination_bucket = self.connection.bucket(
                os.getenv("DESTINATION_BUCKET_NAME")
            )

        if destination_blob_name is None:
            destination_blob_name = blob_name

        blob = self.bucket.blob(blob_name)
        try:
            blob_copy = self.bucket.copy_blob(
                blob, destination_bucket, destination_blob_name
            )
            self.delete_file(blob_name)
            log.info(
                f"Blob {blob_name} has been moved from bucket {self.bucket_name} "
                f"to bucket {destination_bucket} as blob {blob_copy.name}"
            )
        except NotFound as e:
            log.error(f"Blob {blob_name} does not exist")
            raise e

    def upload_from_folder(
        self,
        local_path: str,
        blob_name: str,
    ):
        """
        Upload all files from a local folder to Google Cloud Storage.

        Recursively uploads all files from the specified local directory
        to the GCS bucket, preserving the directory structure. Files are
        uploaded with their relative paths as blob names.

        :param local_path: Path to the local directory to upload
        :type local_path: str
        :param blob_name: Base path/prefix for the uploaded files in the bucket
        :type blob_name: str
        :raises ValueError: If the local path is not a directory

        Example:
            >>> gcs.upload_from_folder("/local/data", "backup/data")
        """
        # Check if the path is a directory
        if not os.path.isdir(local_path):
            raise ValueError("The path is not a directory")

        # Walk through the directory and upload the files
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_path)
                destination_blob_name = os.path.join(blob_name, relative_path).replace(
                    os.path.sep, "/"
                )

                # Upload the file to the bucket
                self.upload_from_filename(local_file_path, destination_blob_name)
