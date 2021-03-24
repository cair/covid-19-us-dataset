import requests
import os
import pandas as pd
import collections


def download(name, url):
    os.makedirs("data", exist_ok=True)
    file_name = f"data/{name}.csv"
    if os.path.exists(file_name):
        return pd.read_csv(file_name)

    s = requests.get(url).content
    with open(file_name, "wb+") as f:
        f.write(s)

    return download(name, url)


def count_common_states(*args):
    full_list = []
    n_items = len(args)

    for l in args:
        full_list.extend(l)

    items = collections.Counter(full_list)
    commons = [k for k, v in items.items() if v == n_items]
    uncommons = [k for k, v in items.items() if v != n_items]

    return commons, uncommons


def remove_uncommon_states(uncommons, *args):
    res = []
    for df in args:
        df = df[~df['state'].isin(uncommons)]
        res.append(df)

    return res


def convert_datestring_to_datetime(*param):
    for df in param:
        df['date'] = pd.to_datetime(df['date'])


def fill_na_by_partial_name(partial, df):
    names = [x for x in df.columns.values if partial in x]
    for name in names:
        df[name] = df[name].fillna(0)



class FileUtils:

    @staticmethod
    def backup_directory(path, backup_prefix):
        import zipfile
        import datetime

        zip_name = f"backup-{backup_prefix}-{str(datetime.datetime.now())}.zip"

        zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(path):
            for file in files:
                zipf.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(path, '..')))
        zipf.close()

        assert os.path.exists(zip_name), "Could not create zip!"

    @staticmethod
    def remove_directory(data):
        import shutil
        dir_path = os.path.dirname(os.path.realpath(__file__))
        full_path = f"{dir_path}/{data}"

        shutil.rmtree(full_path)



