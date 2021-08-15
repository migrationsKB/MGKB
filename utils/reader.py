import os
import json
import gzip


def read_gz_file(filepath):
    """
    read data from gz file.
    :param filepath:
    :return:
    """
    with gzip.open(filepath) as reader:
        data_reader = reader.read()
        data_encoded = data_reader.decode('utf-8')
        file_data = json.loads(data_encoded)
        return file_data


def read_txt_file(filepath):
    """
    read text file into a list of lines.
    :param filepath:
    :return: list of lines.
    """
    with open(filepath) as reader:
        return [line.replace('\n', '') for line in reader.readlines()]


def read_json_file(filepath):
    """
    read data from json file
    :param filepath: file path of the json file
    :return: data
    """
    with open(filepath) as reader:
        return json.load(reader)

if __name__ == '__main__':
    cwd = os.getcwd()
    filepath= os.path.join(cwd, 'data', 'test', 'tweets.txt')
    lines = read_txt_file(filepath)
    print(lines)