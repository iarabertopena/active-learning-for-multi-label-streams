### Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import zipfile
import rarfile
import psutil
import time
import math
import json
import os

from capymoa.stream import NumpyStream
from capymoa.stream import Schema

from IPython.utils import process
from collections import deque
from datetime import datetime
from scipy.io import arff
from pathlib import Path


"""### Datasets"""

PROJECT_DIR = Path("/disk2/iara/projetos/active_learning_multilabel")

DATASET_DIR = PROJECT_DIR / "datasets"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
LOGS_DIR = PROJECT_DIR / "logs"

dataset_paths = {
    "flags": DATASET_DIR / "flags" / "flags.arff",
    "emotions": DATASET_DIR / "emotions" / "emotions.arff",
    "scene": DATASET_DIR / "scene" / "scene.arff",
    "birds": DATASET_DIR / "birds" / "birds.arff",
    "yeast": DATASET_DIR / "Yeast.arff",
    "water-quality": DATASET_DIR / "water-quality.arff",
    "SynHPGrad": DATASET_DIR / "SynHPGrad.arff",
    "SynHPInc": DATASET_DIR / "SynHPInc.arff",
}

n_labels_dict = {
    "flags": 7,
    "emotions": 6,
    "scene": 6,
    "birds": 19,
    "yeast": 14,
    "water-quality": 14,
    "SynHPGrad": 8,
    "SynHPInc": 8
}

def load_multilabel_dataset(dataset_name):

    dataset_path = dataset_paths[dataset_name]

    n_labels = n_labels_dict[dataset_name]

    # carregar arff
    data, meta = arff.loadarff(dataset_path)

    # dataframe
    df = pd.DataFrame(data)

    # converter bytes -> string/int
    for col in df.columns:

        if df[col].dtype == object:

            df[col] = df[col].apply(
                lambda x: x.decode('utf-8')
                if isinstance(x, bytes)
                else x
            )

    # converter tudo para float
    df = df.astype(float)

    # dataset com labels no começo
    if dataset_name in ["SynHPGrad", "SynHPInc", "yeast"]:

        Y = df.iloc[:, :n_labels].values.astype(int)
        X = df.iloc[:, n_labels:].values

    # datasets com labels no final
    else:

        X = df.iloc[:, :-n_labels].values
        Y = df.iloc[:, -n_labels:].values.astype(int)

        # adaptação p/ water-quality: valores > 0 indicam presença da label
        if dataset_name == "water-quality":
            Y = (Y > 0).astype(int)

    return X, Y