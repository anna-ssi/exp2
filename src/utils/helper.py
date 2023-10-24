from os import listdir
from os.path import isfile, join

from typing import List

def get_file_names(path: str, ext: str) -> List[str]:
    """
    Get all file names in a directory
        :param path: path to the directory
        :return: list of file names
    """
    return sorted([join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith(ext)])

