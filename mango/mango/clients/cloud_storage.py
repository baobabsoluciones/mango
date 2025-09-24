from abc import abstractmethod


class CloudStorage:
    """
    Abstract base class for cloud storage operations.

    Defines the interface for cloud storage implementations.
    Concrete subclasses must implement all abstract methods.
    """

    @abstractmethod
    def upload_object(self, contents, blob_name):
        """
        Upload object contents to cloud storage.

        :param contents: The data content to upload
        :param blob_name: Name/path of the blob in cloud storage
        """
        raise NotImplementedError

    @abstractmethod
    def upload_from_filename(self, file_name, blob_name):
        """
        Upload file from local filesystem to cloud storage.

        :param file_name: Path to the local file to upload
        :param blob_name: Name/path of the blob in cloud storage
        """
        raise NotImplementedError
