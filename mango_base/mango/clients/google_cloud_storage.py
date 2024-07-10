import logging as log
import os

from google.api_core.exceptions import NotFound
from google.cloud import storage

from .cloud_storage import CloudStorage


class GoogleCloudStorage(CloudStorage):
    """
    This class handles the connection and some of the most common methods to connect to a Google cloud storage bucket
    and upload, download or delete files
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
        Uploads a python object to the Google cloud bucket with the name set up as blob_name

        :param contents: the python objet to be uploaded. This object has to be able to be converted to a string
            or a bytes type object
        :type contents:
        :param str blob_name: the name the files is going to have once uploaded.
        """
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(contents)
        log.info(f"Uploaded contents to blob {blob_name}")

    def upload_from_filename(self, file_name: str, blob_name: str):
        """
        Uploads a file on the local disk based on the path to the file

        :param str file_name: the path to the file that is going to be uploaded.
        :param str blob_name: the name the files is going to have once uploaded.
        :raises: :class:`FileNotFoundError` if the file does not exist
        """

        if not os.path.exists(file_name):
            log.error(f"The file {file_name} does not exist")
            raise FileNotFoundError(f"The file {file_name} does not exist")

        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(file_name)
        log.info(f"Uploaded file {file_name} to blob {blob_name}")

    def upload_from_file(self, file, blob_name: str):
        """
        Uploads a file from a current open file handle

        :param file: the file handle
        :param str blob_name: the name the files is going to have once uploaded.
        """
        blob = self.bucket.blob(blob_name)
        blob.upload_from_file(file)
        log.info(f"Uploaded file to blob {blob_name}")

    def rename_file(self, blob_name: str, new_name: str):
        """
        Modifies the name of an existing blob

        :param str blob_name: the name of the blob to be renamed
        :param str new_name: the new name given to the blob
        :raises: :class:`NotFound` if the blob does not exist
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
        Downloads a file on the bucket as an object if it can be converted to string or bytes like.

        :param str blob_name: the name of the blob to be downloaded
        :return: the object
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
        Downloads a file on the bucket to the local disk

        :param str blob_name: the name of the blob to be downloaded
        :param str destination_path: the local path where the blob has to be stored
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
        Deletes a file from the bucket

        :param str blob_name: the name of the blob to be deleted
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
        Method to return a list of the files that are stored on the bucket

        :return: a list with the files names on the bucket
        :rtype: list
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
        Copies a file from one bucket to another. If no new name is given it keeps the original name

        :param str blob_name: the name of the blob to be copied
        :param str destination_bucket: the name of the destination bucket
        :param str destination_blob_name: the name of the file on the destination bucket
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
        Moves a blob from one bucket to another. If no new name is given it keeps the original name

        :param str blob_name: the name of the blob to be moved
        :param str destination_bucket: the name of the destination bucket
        :param str destination_blob_name: the name of the blob on the destination bucket
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
        The upload_from_folder function uploads all files from a local folder to the Google Cloud Storage bucket.

        :param str local_path: Specify the local path to the folder
        :param str blob_name: Specify the name of the blob
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
