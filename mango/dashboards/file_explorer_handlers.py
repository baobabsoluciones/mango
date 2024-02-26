import json
import os
from abc import abstractmethod, ABC
from io import BytesIO
from pathlib import Path
from typing import List, Union

from PIL import Image
from google.cloud import storage

from mango.processing import write_json


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
    def get_file_or_folder_paths(self, path: str, element_type: str) -> List[str]:
        pass

    @abstractmethod
    def read_img(self, path: str):
        pass

    @abstractmethod
    def read_markdown(self, path: str):
        pass

    @abstractmethod
    def read_json(self, path: str):
        pass

    @abstractmethod
    def write_json_fe(self, path: str, data: dict):
        pass

    @abstractmethod
    def read_html(self, path: str, encoding: str = "utf-8"):
        pass


class LocalFileExplorerHandler(FileExplorerHandler):
    def is_dir(self, path: str):
        return os.path.isdir(path)

    def path_exists(self, path: str):
        return os.path.exists(path)

    def get_file_or_folder_paths(self, path: str, element_type: str):
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
                    "element_type must be 'file' or 'folder', but got {}".format(
                        element_type
                    )
                )
        return paths

    def read_img(self, path: str):
        return Image.open(path)

    def read_markdown(self, path: str):
        return Path(path).read_text()

    def read_json(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def write_json_fe(self, path: str, data: Union[dict, list]):
        write_json(data, path)

    def read_html(self, path: str, encoding: str = "utf-8"):
        with open(path, encoding=encoding) as f:
            html = f.read()
        return html


class GCPFileExplorerHandler(FileExplorerHandler):

    # Super init
    def __init__(self, path: str, gcp_credentials_path: str):
        super().__init__(path)
        self._gcp_client = storage.Client.from_service_account_json(
            gcp_credentials_path
        )
        self._bucket_name = self._path.split("/")[2]
        self._path = "/".join(self._path.split("/")[3:])
        self._bucket = self._gcp_client.bucket(self._bucket_name)

    def is_dir(self, path: str):
        pass

    def path_exists(self, path: str):
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

        # Append the path to the paths
        paths = [f"gs://{self._bucket_name}/" + path_i for path_i in paths]
        paths.sort()

        return paths

    def read_img(self, path: str):
        blob_image = self._bucket.blob(path.replace(f"gs://{self._bucket_name}/", ""))
        return Image.open(BytesIO(blob_image.download_as_bytes()))

    def read_markdown(self, path: str):
        blob_md = self._bucket.blob(path.replace(f"gs://{self._bucket_name}/", ""))
        return blob_md.download_as_string().decode("utf-8")

    def read_json(self, path: str):
        blob_json = self._bucket.blob(path.replace(f"gs://{self._bucket_name}/", ""))
        json_content = blob_json.download_as_string().decode("utf-8")
        data = json.loads(json_content)
        return data

    def write_json_fe(self, path: str, data: Union[dict, list]):
        blob_json = self._bucket.blob(path.replace(f"gs://{self._bucket_name}/", ""))
        blob_json.upload_from_string(
            data=json.dumps(data, indent=4, sort_keys=False),
            content_type="application/json",
        )

    def read_html(self, path: str, encoding: str = "utf-8"):
        blob_html = self._bucket.blob(path.replace(f"gs://{self._bucket_name}/", ""))
        html = blob_html.download_as_string().decode(encoding)
        return html


if __name__ == "__main__":
    local_path = r"G:\Unidades compartidas\mango\desarrollo\datos\file_explorer_folder"
    local_handler = LocalFileExplorerHandler(local_path)

    print(1)
