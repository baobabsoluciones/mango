from abc import abstractmethod


class CloudStorage:
    @abstractmethod
    def upload_object(self, contents, blob_name):
        raise NotImplementedError

    @abstractmethod
    def upload_from_filename(self, file_name, blob_name):
        raise NotImplementedError
