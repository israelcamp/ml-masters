import json
import pickle
import os


def save_pickle(obj, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(filepath):
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)


def read_json(filepath):
    with open(filepath, 'rb') as file:
        return json.load(file)


def save_json(obj, filepath):
    with open(filepath, 'w') as file:
        json.dump(obj, file)


def create_dir(name):
    if not os.path.exists(name):
        os.mkdir(name)
