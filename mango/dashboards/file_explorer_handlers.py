import os
from abc import abstractmethod, ABC

from google.cloud import storage

gcp_credentials_path = r"C:\Users\GuillermoValle\Documents\codigo\mango\mango\dashboards\.env\cv-apoyos-svc.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_path


class FileExplorerHandler(ABC):
    def __init__(self, path: str):
        self._path = path

    @abstractmethod
    def is_dir(self, path: str):
        pass

    @abstractmethod
    def path_exists(self, path: str):
        pass

    @abstractmethod
    def walk(self, path: str):
        pass


class LocalFileExplorerHandler(FileExplorerHandler):
    def is_dir(self, path: str):
        return os.path.isdir(path)

    def path_exists(self, path: str):
        return os.path.exists(path)

    def walk(self, path: str):
        return os.walk(path)


class GCPFileExplorerHandler(FileExplorerHandler):

    # Super init
    def __init__(self, path: str):
        super().__init__(path)
        self._gcp_client = storage.Client.from_service_account_json(
            gcp_credentials_path
        )
        self._bucket = self._gcp_client.bucket(self._path)
        self._path = f"gs://{path}"

    def is_dir(self, path: str):
        pass

    def path_exists(self, path: str):
        list_files = [
            blob.name
            for blob in self._bucket.list_blobs(prefix=path + "/")
            if blob.name.endswith("/") and len(blob.name.split("/")) == 2
        ]
        return bool(sum(path in file for file in list_files))

    def walk(self, path: str):
        blobs = self._bucket.list_blobs(prefix=path.replace(self._path, "") + "/")

        files = []
        folders = []

        for blob in blobs:
            if blob.name.endswith("/"):
                folders.append(blob.name[len(path) :])
            else:
                files.append(blob.name[len(path) :])

        yield path, folders, files

        for folder in folders:
            yield from self.walk(path + folder + "/")


if __name__ == "__main__":
    local_explorer = LocalFileExplorerHandler(
        r"G:\Unidades compartidas\MisionesIA_VisionArtificialMantenimientoRed\desarrollo\experimentos"
    )
    element_type = "file"
    paths = []
    paths_gcp = []
    for root, dirs, files in local_explorer.walk(local_explorer._path):
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
                "element_type must be 'file' or 'folder', but got {}".format(
                    element_type
                )
            )

    gcp_explorer = GCPFileExplorerHandler("cv-apoyos")
    for root, dirs, files in gcp_explorer.walk("experiments"):
        if element_type == "file":
            for element in files:
                path = os.path.join(root, element)
                paths_gcp.append(path)
        elif element_type == "folder":
            for element in dirs:
                path = os.path.join(root, element)
                paths_gcp.append(path)
        else:
            raise ValueError(
                "element_type must be 'file' or 'folder', but got {}".format(
                    element_type
                )
            )

    print(paths_gcp)
