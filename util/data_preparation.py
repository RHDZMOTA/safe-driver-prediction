import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from conf import settings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DEFAULT_SPLIT = {
    "train": 0.7,
    "test": 0.2,
    "validate": 0.1
}

DEFAULT_TRANSFORMATIONS = {}


class DataSets(object):

    functions = {
        "log": lambda x: np.log(x),
        "root_2": lambda x: np.sqrt(x),
        "root_5": lambda x: np.power(x, 1 / 5),
        "root_7": lambda x: np.power(x, 1 / 7)
    }

    response_function = {
        "identity": lambda x: x,
        "log": lambda x: np.log(x),
        "root_2": lambda x: np.sqrt(x),
        "root_5": lambda x: np.power(x, 1 / 5),
        "root_7": lambda x: np.power(x, 1/7)
    }

    inverse_function = {
        "identity": (lambda x: x),
        "log": lambda x: np.exp(x),
        "root_2": lambda x: np.power(x, 2),
        "root_5": lambda x: np.power(x, 5),
        "root_7": lambda x: np.power(x, 7)
    }

    def __init__(self, data, split_vals=DEFAULT_SPLIT, encode_string=True,
                 one_hot_encode=True, predictive_var="target", link="identity",
                 transformations=DEFAULT_TRANSFORMATIONS):
        data[predictive_var] = self.response_function[link](data[predictive_var].values.astype(np.float))
        for col in transformations:
            data[col] = self.functions[transformations[col]](data[col].values.astype(np.float))
        self.link = link
        self.predictive_var = predictive_var
        self.transformations = transformations
        self.one_hot_encode = one_hot_encode
        self.string_encoder = {}
        self.one_hot_encoder = {}
        self.categorical_cols = None
        self.add_one_here = []
        self.data = data.reset_index(drop=True)
        if encode_string:
            self._encode_string()
        self.n, self.m = data.shape
        self._train_reference = split_vals["train"]
        self._test_validate_reference = split_vals["train"] + split_vals["test"]
        self._split_data()

    def _split_data(self):
        self.train, self.test, self.validate = np.split(
            self.data.sample(frac=1),
            [int(self.n * self._train_reference), int(self.n * self._test_validate_reference)])

    def get_train(self, output_var=False, apply_inverse=False):
        if output_var:
            if apply_inverse:
                return self.inverse_function[self.link](self.train[self.predictive_var].values)
            return self.train[self.predictive_var].values
        return self.train[self.train.columns[self.train.columns != self.predictive_var]]

    def get_test(self, output_var=False, apply_inverse=False):
        if output_var:
            if apply_inverse:
                return self.inverse_function[self.link](self.test[self.predictive_var].values)
            return self.test[self.predictive_var].values
        return self.test[self.test.columns[self.test.columns != self.predictive_var]]

    def get_validate(self, output_var=False, apply_inverse=False):
        if output_var:
            if apply_inverse:
                return self.inverse_function[self.link](self.validate[self.predictive_var].values)
            return self.validate[self.predictive_var].values
        return self.validate[self.validate.columns[self.validate.columns != self.predictive_var]]

    def _encode_string(self):
        self.categorical_cols = list(filter(lambda x: "cat" in x, self.data.columns))
        for col in self.categorical_cols:
            if isinstance(self.data[col].iloc[0], str):
                LE = LabelEncoder()
                LE.fit(sub_data[col])
                classes = [col + "_" + str(i) for i in range(len(LE.classes_))]
                self.string_encoder[col] = LE
                self.data[col] = LE.transform(self.data[col])
            else:
                classes = [col + "_" + str(i) for i in range(len(self.data[col].unique()))]
            if (-1 in self.data[col].unique()) and ("cat" in col):
                self.add_one_here.append(col)
                self.data[col] = self.data[col] + 1
            if self.one_hot_encode:
                OHE = OneHotEncoder()
                OHE.fit(self.data[col].values.reshape(-1, 1))
                vals = OHE.fit_transform(self.data[col].values.reshape(-1, 1)).toarray()
                temp = pd.DataFrame(vals, columns=classes)
                self.data = pd.concat([self.data, temp], axis=1)
                del self.data[col]
                self.one_hot_encoder[col] = OHE

    def external(self, df, output_var=False, apply_inverse=False):
        if output_var:
            if apply_inverse:
                return df[self.predictive_var].values
            return self.response_function[self.link](df[self.predictive_var].values.astype(np.float))
        for col in self.transformations:
            df[col] = self.functions[self.transformations[col]](df[col].values.astype(np.float))
        df = df.reset_index(drop=True)
        for col in self.categorical_cols:
            if col in self.add_one_here:
                df[col] = df[col] + 1
            if isinstance(df[col].iloc[0], str):
                LE = self.string_encoder[col]
                classes = [col + "_" + str(i) for i in range(len(LE.classes_))]
                df[col] = LE.transform(df[col])
            else:
                classes = [col + "_" + str(i) for i in range(len(df[col].unique()))]
            if self.one_hot_encode:
                OHE = self.one_hot_encoder[col]
                vals = OHE.fit_transform(df[col].values.reshape(-1, 1)).toarray()
                temp = pd.DataFrame(vals, columns=classes)
                df = pd.concat([df, temp], axis=1)
                del df[col]
        return df[df.columns[df.columns != self.predictive_var]]


def get_data():
    if os.path.exists("data/datasets/data.pickle"):
        df = pd.read_pickle("data/datasets/data.pickle")
        submit = pd.read_pickle("data/datasets/submit.pickle")
        return df, submit
    raw_df = pd.read_csv(settings.DataFilesConf.FileNames.train)
    submit = pd.read_csv(settings.DataFilesConf.FileNames.test)
    raw_df.index = raw_df.id.values
    submit.index = submit.id.values
    del raw_df["id"]
    del submit["id"]
    raw_df.to_pickle("data/datasets/data.pickle")
    submit.to_pickle("data/datasets/submit.pickle")
    return raw_df, submit


def get_dataset():
    dataset_pickle = settings.DataFilesConf.FileNames.datasets_pickle
    data, submit = get_data()
    if os.path.exists(dataset_pickle):
        with open(dataset_pickle, "rb") as f:
            dump = pickle.load(f)
        return dump, submit
    datasets = DataSets(data,
                        predictive_var="target")
    with open(dataset_pickle, "wb") as f:
        pickle.dump(datasets, f, pickle.HIGHEST_PROTOCOL)
    return datasets, submit