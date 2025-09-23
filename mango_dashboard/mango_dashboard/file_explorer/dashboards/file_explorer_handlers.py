"""
File explorer handlers for managing file operations across different storage systems.

This module provides abstract and concrete implementations for file exploration
and manipulation across local filesystems and Google Cloud Storage. It includes
handlers for reading various file types (images, markdown, JSON, HTML) and
performing directory operations.

Classes:
    FileExplorerHandler: Abstract base class defining the interface for file handlers.
    LocalFileExplorerHandler: Concrete implementation for local filesystem operations.
    GCPFileExplorerHandler: Concrete implementation for Google Cloud Storage operations.
"""

import json
import os
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import List, Union

from google.cloud import storage
from PIL import Image

from mango.processing import write_json


class FileExplorerHandler(ABC):
    """
    Abstract base class for file exploration handlers.

    This class defines the interface that all file handlers must implement.
    It provides abstract methods for common file operations including directory
    checking, path validation, file reading, and JSON writing.

    Attributes:
        _path (str): The base path for file operations.
    """

    def __init__(self, path: str):
        """
        Initialize the file explorer handler.

        :param path: The base path for file operations.
        :type path: str
        """
        self._path = path

    @abstractmethod
    def is_dir(self, path: str):
        """Check if the given path is a directory.

        :param path: The path to check
        :type path: str
        :return: True if the path is a directory, False otherwise
        :rtype: bool
        """
        pass

    @abstractmethod
    def path_exists(self, path: str):
        """Check if the given path exists.

        :param path: The path to check
        :type path: str
        :return: True if the path exists, False otherwise
        :rtype: bool
        """
        pass

    @abstractmethod
    def get_file_or_folder_paths(self, path: str, element_type: str) -> List[str]:
        """Get a list of file or folder paths within the specified directory.

        :param path: The directory path to search in
        :type path: str
        :param element_type: Type of elements to return ('file' or 'folder')
        :type element_type: str
        :return: List of file or folder paths
        :rtype: List[str]
        :raises ValueError: If element_type is not 'file' or 'folder'
        """
        pass

    @abstractmethod
    def read_img(self, path: str):
        """Read an image file from the specified path.

        :param path: The path to the image file
        :type path: str
        :return: The loaded image object
        :rtype: PIL.Image.Image
        """
        pass

    @abstractmethod
    def read_markdown(self, path: str):
        """Read a markdown file from the specified path.

        :param path: The path to the markdown file
        :type path: str
        :return: The content of the markdown file
        :rtype: str
        """
        pass

    @abstractmethod
    def read_json(self, path: str):
        """Read a JSON file from the specified path.

        :param path: The path to the JSON file
        :type path: str
        :return: The parsed JSON data
        :rtype: dict or list
        """
        pass

    @abstractmethod
    def write_json_fe(self, path: str, data: dict):
        """Write data to a JSON file at the specified path.

        :param path: The path where to write the JSON file
        :type path: str
        :param data: The data to write to the JSON file
        :type data: dict
        """
        pass

    @abstractmethod
    def read_html(self, path: str, encoding: str = "utf-8"):
        """Read an HTML file from the specified path.

        :param path: The path to the HTML file
        :type path: str
        :param encoding: The encoding to use
        :type encoding: str
        :return: The content of the HTML file
        :rtype: str
        """
        pass


class LocalFileExplorerHandler(FileExplorerHandler):
    """
    File explorer handler for local filesystem operations.

    This class implements the FileExplorerHandler interface for local filesystem
    operations using standard Python libraries like os and pathlib.
    """

    def is_dir(self, path: str):
        """Check if the given path is a directory using os.path.isdir.

        :param path: The path to check
        :type path: str
        :return: True if the path is a directory, False otherwise
        :rtype: bool
        """
        return os.path.isdir(path)

    def path_exists(self, path: str):
        """Check if the given path exists using os.path.exists.

        :param path: The path to check
        :type path: str
        :return: True if the path exists, False otherwise
        :rtype: bool
        """
        return os.path.exists(path)

    def get_file_or_folder_paths(self, path: str, element_type: str):
        """Get a list of file or folder paths within the specified directory.

        Uses os.walk to recursively traverse the directory and collect paths
        based on the specified element type.

        :param path: The directory path to search in
        :type path: str
        :param element_type: Type of elements to return ('file' or 'folder')
        :type element_type: str
        :return: List of file or folder paths
        :rtype: List[str]
        :raises ValueError: If element_type is not 'file' or 'folder'
        """
        paths = []
        for root, dirs, files in os.walk(path):
            if element_type == "file":
                for element in files:
                    path = os.path.join(root, element)
                    paths.append(path)
            elif element_type == "folder":
                for element in dirs:
                    path = os.path.join(root, element)
                    paths.append(path)
            else:
                raise ValueError(
                    f"element_type must be 'file' or 'folder', but got {element_type}"
                )
        return paths

    def read_img(self, path: str):
        """Read an image file using PIL.Image.open.

        :param path: The path to the image file
        :type path: str
        :return: The loaded image object
        :rtype: PIL.Image.Image
        """
        return Image.open(path)

    def read_markdown(self, path: str):
        """Read a markdown file using pathlib.Path.read_text.

        :param path: The path to the markdown file
        :type path: str
        :return: The content of the markdown file
        :rtype: str
        """
        return Path(path).read_text()

    def read_json(self, path: str):
        """Read a JSON file using json.load.

        :param path: The path to the JSON file
        :type path: str
        :return: The parsed JSON data
        :rtype: dict or list
        """
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def write_json_fe(self, path: str, data: Union[dict, list]):
        """Write data to a JSON file using mango.processing.write_json.

        :param path: The path where to write the JSON file
        :type path: str
        :param data: The data to write to the JSON file
        :type data: Union[dict, list]
        """
        write_json(data, path)

    def read_html(self, path: str, encoding: str = "utf-8"):
        """Read an HTML file with specified encoding.

        :param path: The path to the HTML file
        :type path: str
        :param encoding: The encoding to use
        :type encoding: str
        :return: The content of the HTML file
        :rtype: str
        """
        with open(path, encoding=encoding) as f:
            html = f.read()
        return html


class GCPFileExplorerHandler(FileExplorerHandler):
    """
    File explorer handler for Google Cloud Storage operations.

    This class implements the FileExplorerHandler interface for Google Cloud
    Storage operations using the google-cloud-storage library.

    Attributes:
        _gcp_client (storage.Client): The Google Cloud Storage client.
        _bucket_name (str): The name of the GCS bucket.
        _bucket (storage.bucket.Bucket): The bucket object for operations.
    """

    def __init__(self, path: str, gcp_credentials_path: str):
        """
        Initialize the GCP file explorer handler.

        :param path: The GCS path in format 'gs://bucket-name/path'.
        :type path: str
        :param gcp_credentials_path: Path to the GCP service account JSON file.
        :type gcp_credentials_path: str
        """
        super().__init__(path)
        self._gcp_client = storage.Client.from_service_account_json(
            gcp_credentials_path
        )
        self._bucket_name = self._path.split("/")[2]
        self._path = "/".join(self._path.split("/")[3:])
        self._bucket = self._gcp_client.bucket(self._bucket_name)

    def is_dir(self, path: str):
        """Check if the given path is a directory in GCS.

        This method is not fully implemented for GCS as directories
        are virtual constructs in cloud storage.

        :param path: The path to check
        :type path: str
        :return: Always returns False as this is not implemented
        :rtype: bool
        """
        pass

    def path_exists(self, path: str):
        """Check if the given path exists in GCS.

        Lists blobs with the specified prefix and checks if any match
        the given path.

        :param path: The path to check
        :type path: str
        :return: True if the path exists, False otherwise
        :rtype: bool
        """
        list_files = [
            blob.name
            for blob in self._bucket.list_blobs(
                prefix=path.replace(f"gs://{self._bucket_name}/", "")
            )
        ]
        return bool(
            sum(
                path.replace(f"gs://{self._bucket_name}/", "") in file
                for file in list_files
            )
        )

    def get_file_or_folder_paths(self, path: str, element_type: str = "file"):
        """Get a list of file or folder paths within the specified GCS directory.

        Lists blobs in the bucket with the specified prefix and categorizes
        them as files or folders based on whether they end with '/'.

        :param path: The directory path to search in
        :type path: str
        :param element_type: Type of elements to return
        :type element_type: str
        :return: List of file or folder paths with full GCS URLs
        :rtype: List[str]
        """
        if not path.endswith("/"):
            path += "/"

        blobs = self._bucket.list_blobs(
            prefix=path.replace(f"gs://{self._bucket_name}/", "")
        )

        files = []
        folders = []

        for blob in blobs:
            if blob.name.endswith("/"):
                folders.append(blob.name)
            else:
                files.append(blob.name)

        if element_type == "folder":
            max_depth = max(len(file.split("/")) for file in files)
            for i in range(max_depth):
                for file in files:
                    folder_path = "/".join(file.split("/")[: i + 1])
                    if folder_path in files:
                        continue
                    if not folder_path.endswith("/"):
                        folder_path += "/"
                    folders.append(folder_path)

            paths = list(set(folders))
            paths.remove(path.replace(f"gs://{self._bucket_name}/", ""))
        else:
            paths = files

        paths = [f"gs://{self._bucket_name}/" + path_i for path_i in paths]
        paths.sort()

        return paths

    def read_img(self, path: str):
        """Read an image file from GCS using PIL.Image.open.

        Downloads the blob as bytes and opens it as an image.

        :param path: The GCS path to the image file
        :type path: str
        :return: The loaded image object
        :rtype: PIL.Image.Image
        """
        blob_image = self._bucket.blob(path.replace(f"gs://{self._bucket_name}/", ""))
        return Image.open(BytesIO(blob_image.download_as_bytes()))

    def read_markdown(self, path: str):
        """Read a markdown file from GCS.

        Downloads the blob as a string and decodes it as UTF-8.

        :param path: The GCS path to the markdown file
        :type path: str
        :return: The content of the markdown file
        :rtype: str
        """
        blob_md = self._bucket.blob(path.replace(f"gs://{self._bucket_name}/", ""))
        return blob_md.download_as_string().decode("utf-8")

    def read_json(self, path: str):
        """Read a JSON file from GCS.

        Downloads the blob as a string, decodes it, and parses as JSON.

        :param path: The GCS path to the JSON file
        :type path: str
        :return: The parsed JSON data
        :rtype: dict or list
        """
        blob_json = self._bucket.blob(path.replace(f"gs://{self._bucket_name}/", ""))
        json_content = blob_json.download_as_string().decode("utf-8")
        data = json.loads(json_content)
        return data

    def write_json_fe(self, path: str, data: Union[dict, list]):
        """Write data to a JSON file in GCS.

        Converts the data to JSON string and uploads it to the specified path.

        :param path: The GCS path where to write the JSON file
        :type path: str
        :param data: The data to write to the JSON file
        :type data: Union[dict, list]
        """
        blob_json = self._bucket.blob(path.replace(f"gs://{self._bucket_name}/", ""))
        blob_json.upload_from_string(
            data=json.dumps(data, indent=4, sort_keys=False),
            content_type="application/json",
        )

    def read_html(self, path: str, encoding: str = "utf-8"):
        """Read an HTML file from GCS.

        Downloads the blob as a string and decodes it with the specified encoding.

        :param path: The GCS path to the HTML file
        :type path: str
        :param encoding: The encoding to use
        :type encoding: str
        :return: The content of the HTML file
        :rtype: str
        """
        blob_html = self._bucket.blob(path.replace(f"gs://{self._bucket_name}/", ""))
        html = blob_html.download_as_string().decode(encoding)
        return html
