from os import listdir
from os.path import isfile, join

from typing import List


def get_file_names(path: str, ext: str, keyword: str = None) -> List[str]:
    """
    Get all file names in a directory
        :param path: path to the directory
        :return: list of file names
    """

    paths = []
    for f in listdir(path):
        if isfile(join(path, f)) and f.endswith(ext):
            if keyword is not None and keyword not in f:
                continue
            paths.append(join(path, f))

    return paths