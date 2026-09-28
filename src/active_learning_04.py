### Imports

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
import logging
import zipfile
import rarfile
import psutil
import time
import math
import json
import os

from capymoa.classifier import (AdaptiveRandomForestClassifier)

from capymoa.evaluation import prequential_evaluation

from capymoa.drift.detectors import ADWIN

from capymoa.stream import NumpyStream
from capymoa.stream import Schema

from IPython.utils import process
from collections import deque
from datetime import datetime
from scipy.io import arff
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (hamming_loss, f1_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
'''
def run_elbow_experiment(
    dataset_name,
    k_values=range(2, 21),
    random_state=42,
    batch_size=100
):
    """
    Executa o Elbow Method para escolher o número de clusters (K).

    O experimento utiliza exatamente o mesmo split usado no
    run_experiment():
        - 30% para treinamento inicial
        - 70% para o stream
        - shuffle=False

    O clustering é feito apenas sobre X_train.
    """
    print()
    print(f"ELBOW METHOD | Dataset: {dataset_name}")


    # Carregar dataset
    X, Y = load_multilabel_dataset(dataset_name)

    # Mesmo split do run_experiment()
    X_train, X_stream, Y_train, Y_stream = train_test_split(
        X,
        Y,
        test_size=0.70,
        shuffle=False
    )

    print(
        f"Train: {len(X_train)} | "
        f"Stream: {len(X_stream)}"
    )

    # Testar diferentes valores de K
    results = []

    for k in k_values:

        # Não permite q K seja maior que o número de instâncias disponíveis no treino
        if k > len(X_train):
            print(
                f"K={k} ignorado: "
                f"maior que o número de instâncias de treino."
            )
            continue

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=min(batch_size, len(X_train)),
            n_init=10
        )

        model.fit(X_train)

        inertia = model.inertia_

        results.append({
            "Dataset": dataset_name,
            "K": k,
            "Inertia": inertia
        })

        print(
            f"K={k:2d} | "
            f"Inertia={inertia:.4f}"
        )

    # Transformar resultados em DataFrame
    results_df = pd.DataFrame(results)

    # Criar diretório para resultados do Elbow
    elbow_dir = RESULTS_DIR / "elbow"
    elbow_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Salvar resultados numéricos
    csv_path = elbow_dir / f"{dataset_name}_elbow.csv"

    results_df.to_csv(
        csv_path,
        index=False
    )

    print(f"\nResultados salvos em:")
    print(csv_path)

    # Cria o  gráfico
    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["K"],
        results_df["Inertia"],
        marker="o"
    )

    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Inércia (WCSS)")
    plt.title(
        f"Elbow Method - MiniBatchKMeans - {dataset_name}"
    )

    plt.xticks(results_df["K"])
    plt.grid(True)
    plt.tight_layout()

    # Salva o gráfico
    figure_path = elbow_dir / f"{dataset_name}_elbow.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Gráfico salvo em:")
    print(figure_path)

    return results_df
'''

"""### Métricas de avaliação"""

# métricas
metrics = {
    "f1": "Macro-F1",
    "hamming": "Hamming Loss",
    "exact_match": "Exact Match"
}

"""### Adaptive Threshold"""

class AdaptiveThreshold:

    def __init__(self, block_size, budget):

        self.block_size = block_size
        self.percentile = (1 - budget) * 100

        self.threshold = None

    def initialize(self, scores):

        self.threshold = np.percentile(
            scores,
            self.percentile
        )

        print(
            f"[AdaptiveThreshold] Inicialização | "
            f"n_scores={len(scores)} | "
            f"percentile={self.percentile:.1f} | "
            f"min={np.min(scores):.4f} | "
            f"max={np.max(scores):.4f} | "
            f"threshold={self.threshold:.4f}"
        )

    def update_block(self, scores):
        
        if len(scores) > 0:
            self.threshold = np.percentile(
                scores,
                self.percentile
            )

            print(
                f"[AdaptiveThreshold] Atualização | "
                f"n_scores={len(scores)} | "
                f"percentile={self.percentile:.1f} | "
                f"min={np.min(scores):.4f} | "
                f"max={np.max(scores):.4f} | "
                f"threshold={self.threshold:.4f}"
            )

    def get_threshold(self):

        return self.threshold
'''
class AdaptiveThreshold:

    def __init__(
        self,
        block_size,
        budget,
        method="percentile"
    ):

        self.block_size = block_size
        self.budget = budget
        self.method = method

        self.percentile = (1 - budget) * 100

        self.threshold = None

    def _calculate_threshold(self, scores):

        if self.method == "percentile":

            return np.percentile(
                scores,
                self.percentile
            )

        elif self.method == "mean":

            return np.mean(scores)

        elif self.method == "median":

            return np.median(scores)

        else:

            raise ValueError(
                f"Método de threshold desconhecido: {self.method}"
            )

    def initialize(self, scores):

        self.threshold = self._calculate_threshold(scores)

        print(
            f"[AdaptiveThreshold] Inicialização | "
            f"method={self.method} | "
            f"n_scores={len(scores)} | "
            f"min={np.min(scores):.4f} | "
            f"max={np.max(scores):.4f} | "
            f"threshold={self.threshold:.4f}"
        )

    def update_block(self, scores):

        if len(scores) > 0:

            self.threshold = self._calculate_threshold(scores)

            print(
                f"[AdaptiveThreshold] Atualização | "
                f"method={self.method} | "
                f"n_scores={len(scores)} | "
                f"min={np.min(scores):.4f} | "
                f"max={np.max(scores):.4f} | "
                f"threshold={self.threshold:.4f}"
            )

    def get_threshold(self):

        return self.threshold
'''

"""### Binary Relevance"""

class BinaryRelevance:

    def __init__(self, model_class, n_labels, schema, model_params=None):

        self.n_labels = n_labels
        self.models = []

        if model_params is None:
            model_params = {}

        for _ in range(n_labels):

            model = model_class(
                schema=schema,
                **model_params
            )

            self.models.append(model)

    # previsão
    def predict(self, x):

        predictions = []

        temp_stream = NumpyStream(
             X=np.array([x]),
             y=np.array([0])
          )

        instance = temp_stream.next_instance()

        for model in self.models:

            pred = model.predict(instance)

            if pred is None:
                pred = 0

            predictions.append(int(pred))

        return np.array(predictions)

    # probabilidades
    def predict_proba(self, x):

        probabilities = []

        temp_stream = NumpyStream(
            X=np.array([x]),
            y=np.array([0])
        )

        instance = temp_stream.next_instance()

        for model in self.models:

            proba = model.predict_proba(instance)

            if proba is None:
                proba = np.array([0.5, 0.5])

            probabilities.append(proba)

        return probabilities

    # treino
    def train(self, x, y, label_mask=None):

        # se não tem máscara, assume que todos os rótulos estão disponíveis
        if label_mask is None:
            label_mask = [True] * self.n_labels

        if len(label_mask) != self.n_labels:
            raise ValueError("label_mask deve ter tamanho n_labels")

        for j, model in enumerate(self.models):

            # treina apenas os rótulos observados
            if label_mask[j]:

              temp_stream = NumpyStream(
                  X=np.array([x]),
                  y=np.array([y[j]])
              )

              instance = temp_stream.next_instance()

              model.train(instance)

"""### Active Learning"""

class ActiveLearningStrategy:

    def __init__(self):

        self.total_seen = 0
        self.total_queried = 0

    def query(
        self,
        x=None,
        y_true=None,
        y_pred=None,
        probabilities=None,
        committee_predictions=None,
        committee_probabilities=None
    ):
        raise NotImplementedError

"""#### Random"""

class RandomSampling(ActiveLearningStrategy):

    def __init__(self, budget):

        super().__init__()

        self.budget = budget

    def query(
        self,
        x=None,
        y_true=None,
        y_pred=None,
        probabilities=None,
        committee_predictions=None,
        committee_probabilities=None
    ):

        self.total_seen += 1

        # orçamento esgotado
        if self.total_queried >= self.budget_limit:
            return False

        if np.random.rand() < self.budget:

            self.total_queried += 1

            return True

        return False

"""#### Uncertainty sampling"""

def compute_uncertainty(probabilities):
    uncertainties = []

    for proba in probabilities:
        # Least Confidence
        confidence = np.max(proba)
        uncertainty = 1 - confidence
        uncertainties.append(uncertainty)

    #result = np.mean(uncertainties)
    result = np.max(uncertainties)

    return result

class UncertaintySampling(ActiveLearningStrategy):

    def __init__(self, adaptive_threshold, budget):

        super().__init__()

        self.adaptive_threshold = adaptive_threshold
        self.budget = budget

    def query(
        self,
        x=None,
        y_true=None,
        y_pred=None,
        probabilities=None,
        committee_predictions=None,
        committee_probabilities=None
    ):

        self.total_seen += 1

        # orçamento esgotado
        if self.total_queried >= self.budget_limit:
            return False

        #print(probabilities)
        uncertainty = compute_uncertainty(probabilities)

        threshold = self.adaptive_threshold.get_threshold()
        '''
        print(
            f"Seen={self.total_seen} | "
            f"Threshold={threshold:.4f} | "
            f"Uncertainty={uncertainty:.4f}"
        )
        '''       
        if uncertainty >= threshold:
            self.total_queried += 1
        
            return True

        return False

"""#### Query-by-committee"""

committee_configs = [
    {
        "model_class": AdaptiveRandomForestClassifier,
        "model_params": {"random_seed": 1}
    },
    {
        "model_class": AdaptiveRandomForestClassifier,
        "model_params": {"random_seed": 2}
    },
    {
        "model_class": AdaptiveRandomForestClassifier,
        "model_params": {"random_seed": 3}
    }
]

class Committee:

    def __init__(self, committee_configs, n_labels, schema):

        self.models = []

        for config in committee_configs:

            br_model = BinaryRelevance(
                model_class=config["model_class"],
                n_labels=n_labels,
                schema=schema,
                model_params=config.get("model_params", {})
            )

            self.models.append(br_model)

    def predict_all(self, x):

        predictions = []

        for model in self.models:

            predictions.append(
                model.predict(x)
            )

        return np.array(predictions)

    def predict_proba_all(self, x):

        probabilities = []

        for model in self.models:

            probabilities.append(
                model.predict_proba(x)
            )

        return probabilities

    def train(self, x, y, label_mask=None):

        for model in self.models:

            model.train(x, y, label_mask=label_mask)

def compute_vote_entropy(predictions):

    entropies = []

    n_members = predictions.shape[0]

    for label in range(predictions.shape[1]):

        votes = predictions[:, label]

        counts = np.bincount(
            votes,
            minlength=2
        )

        probs = counts / n_members

        probs = probs[probs > 0]

        entropy = -np.sum(
            probs * np.log2(probs)
        )

        entropies.append(entropy)

    return np.mean(entropies)

class QueryByCommittee(ActiveLearningStrategy):

    def __init__(self, adaptive_threshold, budget, warmup):

        super().__init__()

        self.adaptive_threshold = adaptive_threshold
        self.budget = budget
        self.warmup = warmup

    def query(
        self,
        x=None,
        y_true=None,
        y_pred=None,
        probabilities=None,
        committee_predictions=None,
        committee_probabilities=None
    ):

        self.total_seen += 1

        # orçamento esgotado
        if self.total_queried >= self.budget_limit:
            return False

        # warm-up
        if self.total_seen <= self.warmup:

            self.total_queried += 1
            return True

        if committee_predictions is None:
            return False

        disagreement = compute_vote_entropy(committee_predictions)

        #print(f"Seen={self.total_seen} | "f"Disagreement={disagreement:.4f}")

        threshold = self.adaptive_threshold.get_threshold()
        '''
        print(
            f"Seen={self.total_seen} | "
            f"Threshold={threshold:.10f} | "
            f"Disagreement={disagreement:.10f} | "
            f"Query={disagreement >= threshold}"
        )
        '''
        if disagreement >= threshold:
            self.total_queried += 1

            return True

        return False

"""#### Cluster-Hardness Sampling (CHS)"""

def compute_kdn(x, y, X_history, Y_history, k=5):
    """
    Calcula o kDN de uma instância para cada label.

    Cada label é tratado separadamente como um problema binário.

    kDN_j(x) =
        (1/k) * soma I(y_rj != y_j)

    onde os vizinhos são obtidos a partir de X_history.
    """

    X_history = np.asarray(X_history)
    Y_history = np.asarray(Y_history)
    y = np.asarray(y)

    # Não há histórico suficiente
    if len(X_history) == 0:
        return None

    # Número de vizinhos efetivamente utilizados
    k_actual = min(k, len(X_history))

    # Encontrar os k vizinhos mais próximos
    nn = NearestNeighbors(
        n_neighbors=k_actual
    )

    nn.fit(X_history)

    _, indices = nn.kneighbors(
        np.asarray(x).reshape(1, -1)
    )

    neighbor_indices = indices[0]

    # Labels dos vizinhos
    Y_neighbors = Y_history[neighbor_indices]

    # Discordância para cada label
    disagreements = (
        Y_neighbors != y
    )

    # Média das discordâncias entre os k vizinhos
    kdn_values = np.mean(
        disagreements,
        axis=0
    )

    return kdn_values

def compute_jkdn(x, y, X_history, Y_history, k=5):
    """
    Calcula o Jaccard k-Disagreeing Neighbors (JkDN)
    de uma instância.

    JkDN(x_i) =
        (1/k) * soma [1 - Jaccard(Y_i, Y_j)]

    onde:
        Jaccard(Y_i, Y_j) =
            |Y_i ∩ Y_j| / |Y_i ∪ Y_j|

    Os vizinhos são obtidos a partir de X_history.
    """

    X_history = np.asarray(X_history)
    Y_history = np.asarray(Y_history)
    y = np.asarray(y)

    # Não há histórico suficiente
    if len(X_history) == 0:
        return None

    # Número de vizinhos efetivamente utilizados
    k_actual = min(k, len(X_history))

    # Encontrar os k vizinhos mais próximos
    nn = NearestNeighbors(n_neighbors=k_actual)

    nn.fit(X_history)

    _, indices = nn.kneighbors(np.asarray(x).reshape(1, -1))

    neighbor_indices = indices[0]

    # Labels dos vizinhos
    Y_neighbors = Y_history[neighbor_indices]

    # Calcula a distância de Jaccard para cada vizinho
    jaccard_distances = []

    for y_neighbor in Y_neighbors:

        intersection = np.sum(np.logical_and(y == 1, y_neighbor == 1))

        union = np.sum(np.logical_or(y == 1, y_neighbor == 1))

        # Caso os dois labelsets sejam vazios
        if union == 0:
            jaccard_distance = 0.0

        else:
            jaccard_similarity = intersection / union
            jaccard_distance = (1 - jaccard_similarity)

        jaccard_distances.append(jaccard_distance)

    # Média das distâncias de Jaccard
    jkdn = np.mean(jaccard_distances)

    return jkdn

def compute_cls(y, label_counts, n_instances):
    """
    Calcula o Critical Label Scarcity (CLS) de uma instância.

    CLS(x_i) =
        max_{y_j in Y_i} [-log P(y_j)] / log(N)

    onde:
        P(y_j) = count(y_j) / N

    As estatísticas representam somente as instâncias
    conhecidas antes da instância atual.
    """

    y = np.asarray(y)

    # Não há histórico de labels conhecido
    if n_instances <= 1:
        return None

    # Não há nenhuma label positiva na instância
    positive_labels = np.where(y == 1)[0]

    if len(positive_labels) == 0:
        return 0.0

    cls_values = []

    for label in positive_labels:

        count = label_counts[label]

        # A label presente na instância deveria ter aparecido pelo menos uma vez no histórico.
        if count <= 0:
            continue

        probability = count / n_instances

        cls = (
            -np.log(probability)
            / np.log(n_instances)
        )

        cls_values.append(cls)

    if len(cls_values) == 0:
        return None

    return max(cls_values)

def compute_cb(y, class_counts, n_instances):
    """
    Calcula o Class Balance (CB) para cada label binária.

    CB(x) = P(t(x)) - 1 / |Y|

    No caso per-label, cada label é tratada como um problema
    binário, portanto |Y| = 2.

    Parameters
    ----------
    y : array-like
        Vetor de labels da instância atual (0 ou 1).
    class_counts : ndarray
        Contagem das classes 0 e 1 para cada label.
        Shape: (n_labels, 2).
    n_instances : int
        Número de instâncias utilizadas nas estatísticas.

    Returns
    -------
    cb_values : ndarray or None
        Valor de CB para cada label.
    """
    y = np.asarray(y)

    if n_instances <= 1:
        return None

    cb_values = np.zeros(len(y), dtype=float)

    for label, observed_class in enumerate(y):
        observed_class = int(observed_class)

        count = class_counts[label, observed_class]
        probability = count / n_instances

        cb_values[label] = probability - 0.5
        #cb_values[label] = 0.5 - probability

        print(
            f"CB | label={label} | class={observed_class} | "
            f"p={probability:.4f} | CB={cb_values[label]:.4f}"
        )

    return cb_values

class ClusterManager:

    def __init__(
        self,
        n_clusters,
        random_state=42,
        batch_size=100,
        mode="global",
        n_labels=None
    ):
        if mode not in ["global", "per_label"]:
            raise ValueError("mode deve ser 'global' ou 'per_label'.")

        '''
        if mode == "per_label" and n_labels is None:
            raise ValueError(
                "n_labels deve ser informado quando mode='per_label'."
            )
        '''
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.batch_size = batch_size
        self.mode = mode
        self.n_labels = n_labels

        # modo global: um único clustering para todas as instâncias
        self.model = None

        # modo per_label: um clustering para cada rótulo
        self.models_per_label = None

    def fit(self, X):

        n_clusters = min(self.n_clusters, len(X))
        batch_size = min(self.batch_size, len(X))

        # GLOBAL: um único KMeans
        if self.mode == "global":

            self.model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                batch_size=batch_size,
                n_init=10
            )

            self.model.fit(X)

        # PER_LABEL: um KMeans para cada rótulo
        else:
            self.models_per_label = []

            for label in range(self.n_labels):
                model = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state + label,
                    batch_size=batch_size,
                    n_init=10
                )

                model.fit(X)

                self.models_per_label.append(model)

    def get_cluster(self, x):

        x = np.asarray(x).reshape(1, -1)

        # GLOBAL:
        if self.mode == "global":
            return int(self.model.predict(x)[0])
        
        # PER_LABEL:
        else:
            return [
                int(model.predict(x)[0])
                for model in self.models_per_label
            ]

    def update(self, x):

        x = np.asarray(x).reshape(1, -1)

        # GLOBAL:
        if self.mode == "global":
            self.model.partial_fit(x)

        # PER_LABEL:
        else:
            for model in self.models_per_label:
                model.partial_fit(x)

    @property
    def n_clusters_actual(self):

        # GLOBAL:
        if self.mode == "global":
            if self.model is None:
                return 0

            return self.model.n_clusters
        
        # PER_LABEL:
        else:
            if self.models_per_label is None:
                return 0
            
            return [model.n_clusters for model in self.models_per_label]

class ClusterHardness(ActiveLearningStrategy):

    def __init__(
        self,
        adaptive_threshold,
        budget,
        cluster_manager,
        alpha=0.5
    ):

        super().__init__()

        self.adaptive_threshold = adaptive_threshold
        self.budget = budget
        self.cluster_manager = cluster_manager

        self.alpha = alpha

        self.k = 5

        self.X_history = []
        self.Y_history = []

        # Estatísticas agregadas usadas pelo CLS global
        self.label_counts = None
        self.n_instances = 0

        # Estatísticas usadas pelo CB per-label
        self.class_counts = None
        self.cb_n_instances = 0
        
        '''
        GLOBAL:
        hardness[cluster_id] = H_c

        PER_LABEL:
        hardness[label][cluster_id] = H_{j,c}
        '''

        self.hardness = {}

        # número de observações usadas para cada cluster
        self.cluster_counts = {}

        # Usado no modo global
        self.last_cluster_id = None

        # Usado no modo per_label
        self.last_cluster_ids = None

        # Score usado na última decisão
        self.last_score = None

    def initialize_hardness(
        self,
        x,
        difficulty
    ):

        cluster_info = self.cluster_manager.get_cluster(x)

        # GLOBAL:
        if self.cluster_manager.mode == "global":
            
            cluster_id = cluster_info
            self.update_hardness(cluster_id, difficulty)

        # PER_LABEL:
        else:
            cluster_ids = cluster_info

            for label, cluster_id in enumerate(cluster_ids):
                self.update_hardness(cluster_id, difficulty, label=label)

    def update_hardness(
        self,
        cluster_id,
        difficulty,
        label=None
    ):

        # GLOBAL:
        if self.cluster_manager.mode == "global":
            if cluster_id not in self.hardness:

                self.hardness[cluster_id] = difficulty
                self.cluster_counts[cluster_id] = 1

            else:

                old_hardness = self.hardness[cluster_id]

                self.hardness[cluster_id] = (
                    self.alpha * old_hardness
                    + (1 - self.alpha) * difficulty
                )

                self.cluster_counts[cluster_id] += 1
        
        # PER_LABEL:
        else:
            if label not in self.hardness:
                self.hardness[label] = {}
                self.cluster_counts[label] = {}

            if cluster_id not in self.hardness[label]:

                self.hardness[label][cluster_id] = difficulty
                self.cluster_counts[label][cluster_id] = 1

            else:

                old_hardness = self.hardness[label][cluster_id]

                self.hardness[label][cluster_id] = (
                    self.alpha * old_hardness
                    + (1 - self.alpha) * difficulty
                )

                self.cluster_counts[label][cluster_id] += 1
    
    def update_cls_statistics(self, y_true):
        """
        Atualiza as estatísticas agregadas usadas pelo CLS.

        A atualização ocorre somente depois que o labelset
        verdadeiro da instância já foi obtido.
        """

        y_true = np.asarray(y_true)

        if self.label_counts is None:
            self.label_counts = np.zeros(len(y_true), dtype=int)

        self.label_counts += y_true
        self.n_instances += 1

    def update_cb_statistics(self, y_true):
        y_true = np.asarray(y_true)

        if self.class_counts is None:
            self.class_counts = np.zeros(
                (len(y_true), 2),
                dtype=int
            )

        for label, observed_class in enumerate(y_true):
            self.class_counts[label, int(observed_class)] += 1

        self.cb_n_instances += 1

    def get_hardness(
        self,
        cluster_id,
        label=None
    ):
        # GLOBAL:
        if self.cluster_manager.mode == "global":

            # cluster ainda não observado
            if cluster_id not in self.hardness:
                return 0.0

            return self.hardness[cluster_id]
        
        # PER_LABEL:
        else:
            # label ainda não observado
            if label not in self.hardness:
                return 0.0

            # cluster ainda não observado para este label
            if cluster_id not in self.hardness[label]:
                return 0.0

            return self.hardness[label][cluster_id]

    def get_score(self, cluster_info):

        # GLOBAL:
        if self.cluster_manager.mode == "global":

            return self.get_hardness(cluster_info)


        # PER_LABEL:
        else:
            cluster_ids = cluster_info

            hardness_values = []

            for label, cluster_id in enumerate(cluster_ids):

                hardness = self.get_hardness(
                    cluster_id,
                    label=label
                )

                hardness_values.append(hardness)

            return np.mean(hardness_values)

    def query(
        self,
        x=None,
        y_true=None,
        y_pred=None,
        probabilities=None,
        committee_predictions=None,
        committee_probabilities=None
    ):

        self.total_seen += 1

        # orçamento esgotado
        if self.total_queried >= self.budget_limit:
            return False

        # identifica o cluster antes de atualizar o clustering
        cluster_info = self.cluster_manager.get_cluster(x)

        # calcula o score de acordo com o modo
        score = self.get_score(cluster_info)

        # salva o score usado na decisão
        self.last_score = score

        # GLOBAL:
        if self.cluster_manager.mode == "global":

            # guarda o cluster usado na decisão
            self.last_cluster_id = cluster_info
        
            print(
                f"Seen={self.total_seen} | "
                f"Cluster={cluster_info} | "
                f"Hardness={score:.10f} | "
                f"Threshold={self.adaptive_threshold.get_threshold():.10f}"
            )
        
        # PER_LABEL:
        else:

            # guarda os clusters usados na decisão
            self.last_cluster_ids = cluster_info

            print(
                f"Seen={self.total_seen} | "
                f"Clusters={cluster_info} | "
                f"MeanHardness={score:.10f} | "
                f"Threshold={self.adaptive_threshold.get_threshold():.10f}"
            )

        # atualiza o clustering usando somente X
        self.cluster_manager.update(x)

        # threshold atual
        threshold = self.adaptive_threshold.get_threshold()

        if score >= threshold:

            self.total_queried += 1

            return True

        return False

    def update_after_query(
        self,
        x,
        y_true,
        #y_pred
    ):

        # GLOBAL:
        if self.cluster_manager.mode == "global":

            # Calcula o CLS
            cls = compute_cls(
                y=y_true,
                label_counts=self.label_counts,
                n_instances=self.n_instances
            )

            if cls is None:
                return

            cluster_id = self.last_cluster_id

            self.update_hardness(
                cluster_id,
                cls
            )

            # Somente depois de calcular o CLS, incorpora a instância às estatísticas conhecidas
            self.update_cls_statistics(y_true)

        # PER_LABEL:
        else:

            # Calcula o CB
            cb_values = compute_cb(
                y=y_true,
                class_counts=self.class_counts,
                n_instances=self.cb_n_instances
            )

            if cb_values is None:
                return

            cluster_ids = self.last_cluster_ids

            for label, cluster_id in enumerate(cluster_ids):
                self.update_hardness(
                    cluster_id,
                    cb_values[label],
                    label=label
                )

            # Somente depois de calcular o CB, incorpora a instância às estatísticas conhecidas
            self.update_cb_statistics(y_true)

        # Depois de calcular a métrica, a instância passa a fazer parte do histórico conhecido
        self.X_history.append(x)
        self.Y_history.append(y_true)

"""#### Cluster-Hardness + Uncertainty Sampling"""

class ClusterHardnessUncertainty(ClusterHardness):

    def __init__(
        self,
        adaptive_threshold,
        budget,
        cluster_manager,
        alpha=0.5,
        uncertainty_weight=0.5
    ):

        super().__init__(
            adaptive_threshold,
            budget,
            cluster_manager,
            alpha
        )

        self.uncertainty_weight = uncertainty_weight

        self.last_hardness = None
        self.last_uncertainty = None
        self.last_score = None
    
    def compute_score(self, hardness, uncertainty):

        return (
            (1 - self.uncertainty_weight) * hardness
            + self.uncertainty_weight * uncertainty
        )

    def query(
        self,
        x=None,
        y_true=None,
        y_pred=None,
        probabilities=None,
        committee_predictions=None,
        committee_probabilities=None
    ):
    
        self.total_seen += 1

        # orçamento esgotado
        if self.total_queried >= self.budget_limit:
            return False

        # identifica o cluster antes de atualizar o clustering
        cluster_info = self.cluster_manager.get_cluster(x)

        # calcula o score de acordo com o modo
        hardness = self.get_score(cluster_info)

        # calcula a incerteza da instância
        uncertainty = compute_uncertainty(probabilities)

        # combina os scores
        score = self.compute_score(hardness, uncertainty)

        # salva os scores usados na decisão
        self.last_hardness = hardness
        self.last_uncertainty = uncertainty
        self.last_score = score

        # GLOBAL:
        if self.cluster_manager.mode == "global":
            
            # guarda o cluster usado na decisão
            self.last_cluster_id = cluster_info
            '''
            print(
                f"Seen={self.total_seen} | "
                f"Clusters={cluster_info} | "
                f"MeanHardness={hardness:.10f} | "
                f"Uncertainty={uncertainty:.10f} | "
                f"Score={score:.10f} | "
                f"Threshold={self.adaptive_threshold.get_threshold():.10f}"
            )
        '''
        # PER_LABEL:
        else:
            
            # guarda os clusters usados na decisão
            self.last_cluster_ids = cluster_info
            '''
            print(
                f"Seen={self.total_seen} | "
                f"Clusters={cluster_info} | "
                f"MeanHardness={hardness:.10f} | "
                f"Uncertainty={uncertainty:.10f} | "
                f"Score={score:.10f} | "
                f"Threshold={self.adaptive_threshold.get_threshold():.10f}"
            )
            '''
        # atualiza o clustering usando somente X
        self.cluster_manager.update(x)

        # threshold atual
        threshold = self.adaptive_threshold.get_threshold()

        if score >= threshold:

            self.total_queried += 1

            return True

        return False

"""#### Bounds"""

"""##### Lower Bound -> sem consultas (budget = 0%)"""

class NoQuery(ActiveLearningStrategy):

    def __init__(self):

        super().__init__()

        self.budget = 0

    def query(
        self,
        x=None,
        y_true=None,
        y_pred=None,
        probabilities=None,
        committee_predictions=None,
        committee_probabilities=None
    ):

        self.total_seen += 1

        return False

"""##### Upper Bound -> aprendizado totalmente supervisionado (budget = 100%)"""

class FullSupervision(ActiveLearningStrategy):

    def __init__(self):

        super().__init__()

        self.budget = 1.0

    def query(
        self,
        x=None,
        y_true=None,
        y_pred=None,
        probabilities=None,
        committee_predictions=None,
        committee_probabilities=None
    ):

        self.total_seen += 1
        self.total_queried += 1

        return True

"""### Pool-based Active Learning"""

class Pool:

    def __init__(self, pool_size):

        self.pool_size = pool_size
        self.instances = []

    def add(
        self,
        x,
        y,
        y_pred,
        probabilities,
        committee_predictions=None,
        committee_probabilities=None,
        stream_index=None
    ):

        self.instances.append({
            "x": x,
            "y": y,
            "y_pred": y_pred,
            "probabilities": probabilities,
            "committee_predictions": committee_predictions,
            "committee_probabilities": committee_probabilities,
            "stream_index": stream_index
        })

    def is_full(self):

        return len(self.instances) >= self.pool_size

    def clear(self):

        self.instances = []

    def __len__(self):

        return len(self.instances)

class PoolBasedStrategy(ActiveLearningStrategy):

    def __init__(self, budget):

        super().__init__()

        self.budget = budget

    def query_pool(self, pool):

        raise NotImplementedError

"""#### Random"""

class PoolRandomSampling(PoolBasedStrategy):

    def __init__(self, budget):

        super().__init__(budget)

    def query_pool(self, pool):

        remaining_budget = self.budget_limit - self.total_queried

        if remaining_budget <= 0:
            return []

        k = min(
            self.pool_budget,
            remaining_budget,
            len(pool.instances)
        )

        indices = np.random.choice(
            len(pool.instances),
            size=k,
            replace=False
        )

        self.total_seen += len(pool.instances)
        self.total_queried += len(indices)

        return indices

"""#### Uncertainty sampling"""

class PoolUncertaintySampling(PoolBasedStrategy):

    def __init__(self, budget):

        super().__init__(budget)

    def query_pool(self, pool):

        remaining_budget = self.budget_limit - self.total_queried

        if remaining_budget <= 0:
            return []

        scores = []

        for inst in pool.instances:

            u = compute_uncertainty(
                inst["probabilities"]
            )

            scores.append(u)

        order = np.argsort(scores)[::-1]

        k = min(
            self.pool_budget,
            remaining_budget,
            len(order)
        )

        selected = order[:k]

        self.total_seen += len(pool.instances)
        self.total_queried += len(selected)

        return selected

"""#### Query-by-committee"""

class PoolQueryByCommittee(PoolBasedStrategy):

    def __init__(self, budget):

        super().__init__(budget)

    def query_pool(self, pool):

        remaining_budget = (
            self.budget_limit - self.total_queried
        )

        if remaining_budget <= 0:
            return []

        scores = []

        for inst in pool.instances:

            disagreement = compute_vote_entropy(
                inst["committee_predictions"]
            )

            scores.append(disagreement)

        # ordenar da maior discordância para a menor
        order = np.argsort(scores)[::-1]

        k = min(
            self.pool_budget,
            remaining_budget,
            len(order)
        )

        selected = order[:k]

        self.total_seen += len(pool.instances)
        self.total_queried += len(selected)

        return selected

"""### Pool Cluster-Hardness Sampling (CHS)"""

class PoolClusterHardness(PoolBasedStrategy):

    def __init__(
        self,
        budget,
        cluster_manager,
        hardness,
        alpha=0.5
    ):
        super().__init__(budget)

        self.cluster_manager = cluster_manager
        self.hardness = hardness

        self.k = 5

        self.alpha = alpha

        # Histórico usado para calcular kDN / JkDN
        self.X_history = []
        self.Y_history = []

        # Estatísticas usadas pelo CLS global
        self.label_counts = None
        self.n_instances = 0

        # Estatísticas usadas pelo CB per-label
        self.class_counts = None
        self.cb_n_instances = 0

    def update_hardness(
        self,
        cluster_id,
        difficulty,
        label=None
    ):
        if self.cluster_manager.mode == "global":

            if cluster_id not in self.hardness:
                self.hardness[cluster_id] = difficulty

            else:
                old_hardness = self.hardness[cluster_id]

                self.hardness[cluster_id] = (
                    self.alpha * old_hardness
                    + (1 - self.alpha) * difficulty
                )

        else:

            if label not in self.hardness:
                self.hardness[label] = {}

            if cluster_id not in self.hardness[label]:
                self.hardness[label][cluster_id] = difficulty

            else:
                old_hardness = self.hardness[label][cluster_id]

                self.hardness[label][cluster_id] = (
                    self.alpha * old_hardness
                    + (1 - self.alpha) * difficulty
                )

    def get_hardness(
        self,
        cluster_id,
        label=None
    ):

        if self.cluster_manager.mode == "global":

            if cluster_id not in self.hardness:
                return 0.0

            return self.hardness[cluster_id]

        else:

            if label not in self.hardness:
                return 0.0

            if cluster_id not in self.hardness[label]:
                return 0.0

            return self.hardness[label][cluster_id]

    def get_score(self, x):

        cluster_info = self.cluster_manager.get_cluster(x)

        # GLOBAL
        if self.cluster_manager.mode == "global":

            return self.get_hardness(cluster_info)

        # PER_LABEL
        else:

            hardness_values = []

            for label, cluster_id in enumerate(cluster_info):

                hardness = self.get_hardness(
                    cluster_id,
                    label=label
                )

                hardness_values.append(hardness)

            return np.mean(hardness_values)

    def update_after_query(
        self,
        x,
        y_true
    ):
        print("\n--- UPDATE POOL CHS ---")
        print("Modo:", self.cluster_manager.mode)
        print("y_true:", y_true)

        if self.cluster_manager.mode == "global":
            print("Antes:")
            print("  n_instances:", self.n_instances)
            print("  label_counts:", self.label_counts)

        else:
            print("Antes:")
            print("  cb_n_instances:", self.cb_n_instances)
            print("  class_counts:\n", self.class_counts)

        if len(self.X_history) == 0:
            self.X_history.append(x)
            self.Y_history.append(y_true)
            return

        # GLOBAL -> CLS
        if self.cluster_manager.mode == "global":

            cls = compute_cls(
                y=y_true,
                label_counts=self.label_counts,
                n_instances=self.n_instances
            )

            if cls is not None:

                cluster_id = self.cluster_manager.get_cluster(x)

                self.update_hardness(
                    cluster_id,
                    cls
                )

            # Atualiza as estatísticas somente depois de calcular a dificuldade da instância
            self.update_cls_statistics(y_true)

        # PER_LABEL -> CB
        else:

            cb_values = compute_cb(
                y=y_true,
                class_counts=self.class_counts,
                n_instances=self.cb_n_instances
            )

            if cb_values is not None:

                cluster_ids = self.cluster_manager.get_cluster(x)

                for label, cluster_id in enumerate(cluster_ids):
                    # CB original: P(t(x)) - 0.5; invertido para alinhar maior valor à maior hardness
                    hardness = -cb_values[label] 

                    self.update_hardness(
                        cluster_id,
                        cb_values[label],
                        label=label
                    )

            # Atualiza as estatísticas somente depois de calcular a dificuldade da instância
            self.update_cb_statistics(y_true)

            if self.cluster_manager.mode == "global":

                print("Depois:")
                print("  n_instances:", self.n_instances)
                print("  label_counts:", self.label_counts)

            else:

                print("Depois:")
                print("  cb_n_instances:", self.cb_n_instances)
                print("  class_counts:\n", self.class_counts)

            print("Hardness:", self.hardness)
            print("------------------------")
    
        # Adiciona a instância ao histórico
        # (mantem o histórico por enquanto, para preservar a estrutura original)
        self.X_history.append(x)
        self.Y_history.append(y_true)

    def update_cls_statistics(self, y_true):

        for label, value in enumerate(y_true):
            if value == 1:
                self.label_counts[label] += 1

        self.n_instances += 1

    def update_cb_statistics(self, y_true):

        for label, value in enumerate(y_true):
            self.class_counts[label, int(value)] += 1

        self.cb_n_instances += 1

    def query_pool(self, pool):

        remaining_budget = (
            self.budget_limit - self.total_queried
        )

        if remaining_budget <= 0:
            return []

        scores = []

        for inst in pool.instances:

            score = self.get_score(
                inst["x"]
            )

            scores.append(score)

        # maior hardness primeiro
        order = np.argsort(scores)[::-1]

        k = min(
            self.pool_budget,
            remaining_budget,
            len(order)
        )

        selected = order[:k]

        self.total_seen += len(pool.instances)
        self.total_queried += len(selected)

        return selected

"""#### Pool Cluster-Hardness + Uncertainty Sampling"""

class PoolClusterHardnessUncertainty(PoolClusterHardness):

    def __init__(
        self,
        budget,
        cluster_manager,
        hardness,
        alpha=0.5,
        uncertainty_weight=0.5
    ):

        super().__init__(
            budget=budget,
            cluster_manager=cluster_manager,
            hardness=hardness,
            alpha=alpha
        )

        self.uncertainty_weight = uncertainty_weight

    def query_pool(self, pool):

        remaining_budget = (
            self.budget_limit - self.total_queried
        )

        if remaining_budget <= 0:
            return []

        scores = []

        for inst in pool.instances:

            # Hardness do CHS
            hardness = self.get_score(
                inst["x"]
            )

            # Uncertainty Sampling
            uncertainty_score = compute_uncertainty(
                inst["probabilities"]
            )

            # Score combinado
            score = (
                (1 - self.uncertainty_weight) * hardness
                + self.uncertainty_weight * uncertainty_score
            )

            scores.append(score)

        # maior score primeiro
        order = np.argsort(scores)[::-1]

        k = min(
            self.pool_budget,
            remaining_budget,
            len(order)
        )

        selected = order[:k]

        self.total_seen += len(pool.instances)
        self.total_queried += len(selected)

        return selected

"""### Cálculo de budget por pool"""

def compute_pool_budget(strategy, n_instances, pool_size):

    strategy.budget_limit = int(strategy.budget * n_instances)

    n_pools = math.ceil(n_instances / pool_size)

    strategy.pool_budget = max(1, math.ceil(strategy.budget_limit / n_pools))

    print(f"Budget total: {strategy.budget_limit}")
    print(f"Number of pools: {n_pools}")
    print(f"Budget per pool: {strategy.pool_budget}")

"""### Salvar resultados"""

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULTS_DIR = Path(
    f"/disk2/iara/projetos/active_learning_multilabel/results/run_{RUN_TIMESTAMP}"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_result(
    dataset,
    strategy,
    budget=None,
    mode="online",
    results=None,
    output_dir=RESULTS_DIR
):

    # Salva um experimento em um CSV.
    # Se o arquivo já existir, adiciona uma nova linha.

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{mode}_results.csv"

    row = {
        "Dataset": dataset,
        "Strategy": strategy,
        "Budget": budget,
        **{k: v for k, v in results.items() if k != "History"}
    }

    df = pd.DataFrame([row])

    if output_file.exists():
        df.to_csv(output_file, mode="a", header=False, index=False)
    else:
        df.to_csv(output_file, index=False)

def save_history(
    history,
    dataset_name,
    strategy_name,
    mode,
    output_dir=RESULTS_DIR,
    budget=None,
    repetitions=None
):

    history_dir = Path(output_dir) / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    if budget is None:
        filename = f"{dataset_name}_{strategy_name}_{mode}.json"
    else:
        filename = (
            f"{dataset_name}_{strategy_name}_{int(budget*100)}_{mode}.json"
        )

    history_data = {
        "dataset": dataset_name,
        "strategy": strategy_name,
        "mode": mode,
        "budget": budget,
        **history
    }

    if repetitions is not None:
        history_data["repetitions"] = repetitions

    with open(history_dir / filename, "w") as f:
        json.dump(history_data, f, indent=4)

### Logs
logging.basicConfig(
    filename=LOGS_DIR / "experiment.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

error_logger = logging.getLogger("errors")
error_handler = logging.FileHandler("logs/errors.log")
error_handler.setLevel(logging.ERROR)

error_logger.addHandler(error_handler)

"""### Cálculo da média de múltiplas execuções do Random."""
def average_results(results_list):

    averaged = {}

    # Métricas numéricas
    metrics = [
        "Hamming Loss",
        "Exact Match",
        "F1-score",
        "Execution Time (s)",
        "Memory Usage (MB)",
        "Queried Instances",
        "Queries per Second",
        "Query Rate"
    ]

    for metric in metrics:
        averaged[metric] = np.mean(
            [r[metric] for r in results_list]
        )

    # Número de drifts
    averaged["Number of Drifts"] = np.mean(
        [r["Number of Drifts"] for r in results_list]
    )

    # Não existe uma média bem definida para os pontos de drift
    averaged["Drift Points"] = []

    # Média do histórico
    averaged_history = {}

    # Componentes que são iguais em todas as execuções
    averaged_history["progress"] = results_list[0]["History"]["progress"]
    
    # Média das posições e índices de query
    averaged_history["query_positions"] = (results_list[0]["History"]["query_positions"])
    averaged_history["query_indices"] = (results_list[0]["History"]["query_indices"])

    # Média das curvas
    for metric in ["hamming", "exact_match", "f1", "queries"]:

        values = np.array([
            r["History"][metric]
            for r in results_list
        ])

        averaged_history[metric] = np.mean(
            values,
            axis=0
        ).tolist()

    averaged["History"] = averaged_history

    return averaged

def save_random_queries(
    query_runs,
    dataset_name,
    budget,
    output_dir=RESULTS_DIR
):

    queries_dir = Path(output_dir) / "random_queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "dataset": dataset_name,
        "strategy": "Random",
        "mode": "online",
        "budget": budget,
        "repetitions": len(query_runs),
        "runs": query_runs
    }

    filename = (
        f"{dataset_name}_Random_{int(budget * 100)}_queries.json"
    )

    with open(queries_dir / filename, "w") as f:
        json.dump(data, f, indent=4)
'''
"""### Elbow Method"""
k_values = range(2, 21)

elbow_results = {}

for dataset_name in dataset_paths.keys():

    elbow_results[dataset_name] = run_elbow_experiment(
        dataset_name=dataset_name,
        k_values=k_values,
        random_state=42,
        batch_size=100
    )
'''
"""### Loop principal"""
def run_experiment(
    dataset_name,
    strategy,
    model_class,
    mode="online",
    pool_size=None,
    block_size=None,
    random_seed=None,
    n_clusters=10
):

    print(f"Dataset: {dataset_name}")

    if random_seed is not None:
        np.random.seed(random_seed)

    # carregar dados
    X, Y = load_multilabel_dataset(dataset_name)

    if isinstance(strategy, ClusterHardness):
        strategy.cluster_manager.n_labels = Y.shape[1]

    # split (para teste inicial)
    X_train, X_stream, Y_train, Y_stream = train_test_split(
        X,
        Y,
        test_size=0.70,
        shuffle=False
    )

    print(f"Train: {len(X_train)} | "f"Stream: {len(X_stream)}")

    n = len(X_stream)

    if block_size is None:
        block_size = min(100, n)

    # cálculo do orçamento
    if hasattr(strategy, "budget"):
        if mode == "pool":
            compute_pool_budget(strategy, n_instances=n, pool_size=pool_size)

        else:
            strategy.budget_limit = int(strategy.budget * n)

            print(f"Budget total: {strategy.budget_limit}")

    # checkpoints em %
    checkpoints = [
        max(1, round(n * pct / 100))
        for pct in range(10, 101, 10)
    ]

    strategy.total_seen = 0
    strategy.total_queried = 0

    # schema
    dummy_X = np.zeros((1, X.shape[1]))
    dummy_y = np.zeros(1)

    stream = NumpyStream(
        X=dummy_X,
        y=dummy_y
    )

    schema = stream.get_schema()

    # modelo principal
    model = BinaryRelevance(
        model_class=model_class,
        n_labels=Y.shape[1],
        schema=schema
    )

    # detector de drift
    adwin = ADWIN()

    drift_points = []

    # comitê para QBC
    committee = None

    if isinstance(strategy, (QueryByCommittee, PoolQueryByCommittee)):
        committee = Committee(
            committee_configs=committee_configs,
            n_labels=Y.shape[1],
            schema=schema
        )

    uses_probabilities = isinstance(
        strategy,
        (
            UncertaintySampling, 
            PoolUncertaintySampling,
            ClusterHardnessUncertainty,
            PoolClusterHardnessUncertainty
        )
    )

    uses_committee = isinstance(
        strategy,
        (QueryByCommittee, PoolQueryByCommittee)
    )

    uses_cluster_hardness = isinstance(
        strategy,
        (ClusterHardness, PoolClusterHardness)
    )

    # ClusterHardness -> inicializa clusters antes de aprender a instância
    if uses_cluster_hardness:
        if strategy.cluster_manager.mode == "per_label":
            strategy.cluster_manager.n_labels = Y_train.shape[1]
        
        strategy.cluster_manager.fit(X_train)

        if uses_cluster_hardness:
            strategy.X_history = []
            strategy.Y_history = []

             # Estatísticas iniciais para o CLS
            strategy.label_counts = np.zeros(Y_train.shape[1], dtype=int)
            strategy.n_instances = 0

            # Estatísticas iniciais para o CB
            strategy.class_counts = np.zeros((Y_train.shape[1], 2), dtype=int)
            strategy.cb_n_instances = 0

            for i in range(len(X_train)):
                x_train = X_train[i]
                y_train = Y_train[i]

                # Remove a própria instância do conjunto de vizinhos
                X_history = np.delete(X_train, i, axis=0)
                Y_history = np.delete(Y_train, i, axis=0)

                # Obtém o(s) cluster(s) da instância
                cluster_info = strategy.cluster_manager.get_cluster(x_train)

                # GLOBAL: calcula o CLS para a instância
                if strategy.cluster_manager.mode == "global":
                    cls = compute_cls(
                        y=y_train,
                        label_counts=strategy.label_counts,
                        n_instances=strategy.n_instances
                    )

                    if cls is not None:
                        '''
                        for cluster_id in cluster_info:
                            strategy.update_hardness(
                                cluster_id,
                                cls
                            )
                        '''
                    
                        strategy.update_hardness(
                            cluster_info,
                            cls
                        )

                    strategy.update_cls_statistics(y_train)

                # PER_LABEL: calcula o CB para cada label da instância
                else:
                    cb_values = compute_cb(
                        y=y_train,
                        class_counts=strategy.class_counts,
                        n_instances=strategy.cb_n_instances
                    )

                    if cb_values is not None:
                        for label, cluster_id in enumerate(cluster_info):

                            strategy.update_hardness(
                                cluster_id,
                                cb_values[label],
                                label=label
                            )
                        
                    strategy.update_cb_statistics(y_train)


            # Todo o conjunto inicial passa a ser o histórico conhecido
            strategy.X_history = list(X_train)
            strategy.Y_history = list(Y_train)

        print(
            f"Clusters inicializados: "
            f"{strategy.cluster_manager.n_clusters_actual}"
        )

    # treinamento inicial (_% do dataset)
    for i in range(len(X_train)):

        x_train = X_train[i]
        y_train = Y_train[i]

        #if mode == "pool" and isinstance(strategy, PoolClusterHardness):
            #print("DEBUG POOL:", type(x_train), np.shape(x_train), type(y_train), np.shape(y_train), y_train)

        # aprende depois da avaliação
        model.train(
            x_train,
            y_train
        )

        if committee is not None:

            committee.train(
                x_train,
                y_train
            )

    # inicialização do threshold adaptativo
    if mode == "online" and isinstance(strategy, (UncertaintySampling, ClusterHardnessUncertainty)):

        scores = []

        for x in X_train:

            probabilities = model.predict_proba(x)

            score = compute_uncertainty(probabilities)

            scores.append(score)

        strategy.adaptive_threshold.initialize(scores)
    
    if mode == "online" and isinstance(strategy, QueryByCommittee):

        scores = []

        for x in X_train:

            predictions = committee.predict_all(x)

            score = compute_vote_entropy(predictions)

            scores.append(score)

        strategy.adaptive_threshold.initialize(scores)
    
    if mode == "online" and isinstance(strategy, ClusterHardnessUncertainty):
        scores = []

        for x in X_train:
            cluster_info = strategy.cluster_manager.get_cluster(x)
            hardness = strategy.get_score(cluster_info)

            probabilities = model.predict_proba(x)
            uncertainty = compute_uncertainty(probabilities)

            score = strategy.compute_score(hardness, uncertainty)
            scores.append(score)

        if len(scores) > 0:
            strategy.adaptive_threshold.initialize(scores)

            print(
                "ClusterHardness + Uncertainty threshold inicial:",
                strategy.adaptive_threshold.get_threshold()
            )

    elif mode == "online" and isinstance(strategy, ClusterHardness):
        if strategy.cluster_manager.mode == "global":
            scores = list(strategy.hardness.values())
        else:
            scores = []
            for label_hardness in strategy.hardness.values():
                scores.extend(label_hardness.values())

        if len(scores) > 0:
            strategy.adaptive_threshold.initialize(scores)

            print(
                "ClusterHardness threshold inicial:",
                strategy.adaptive_threshold.get_threshold()
            )

    # tempo inicial
    start_time = time.time()

    # memória inicial
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024**2

    # métricas
    hamming_scores = []
    exact_match_scores = []
    f1_scores = []

    # histórico opara gráficos
    history = {
        "progress": [],
        "hamming": [],
        "exact_match": [],
        "f1": [],
        "queries": [],
        "query_positions": [],
        "query_indices": [],
        "evaluated": []
    }

    # pool (apenas para modo pool)
    if mode == "pool":
        pool = Pool(pool_size)

    # armazena scores usados para threshold adaptativo
    block_scores = []

    # loop prequential (70% do dataset)
    for i in range(len(X_stream)):
        x = X_stream[i]
        y = Y_stream[i]

        # previsão do modelo principal
        y_pred = model.predict(x)

        # probabilidades do modelo principal
        probs = None

        if uses_probabilities:
            probs = model.predict_proba(x)

            if (
                mode == "online"
                and isinstance(strategy, UncertaintySampling)
            ):
                score = compute_uncertainty(probs)
                block_scores.append(score)

        # previsões do comitê
        committee_predictions = None

        if uses_committee:
            committee_predictions = committee.predict_all(x)

            if mode == "online":
                score = compute_vote_entropy(committee_predictions)
                block_scores.append(score)
        
        # métricas
        ham = hamming_loss(y, y_pred)

        exact = int(np.array_equal(y, y_pred))

        f1 = f1_score(
            y,
            y_pred,
            average="macro",
            zero_division=0
        )

        hamming_scores.append(ham)
        exact_match_scores.append(exact)
        f1_scores.append(f1)

        # detector de drift
        adwin.add_element(ham)

        if adwin.detected_change():
            drift_points.append(i)

        # salvar histórico nos checkpoints
        if (i + 1) in checkpoints:
            history["progress"].append(round(100 * (i + 1) / n))

            if len(hamming_scores) > 0:
                history["hamming"].append(np.mean(hamming_scores))
                history["exact_match"].append(np.mean(exact_match_scores))
                history["f1"].append(np.mean(f1_scores))

            else:
                history["hamming"].append(np.nan)
                history["exact_match"].append(np.nan)
                history["f1"].append(np.nan)

            history["queries"].append(strategy.total_queried)
            history["evaluated"].append(len(hamming_scores))

        # Active Learning
        # ONLINE
        if mode == "online":
            queried = strategy.query(
                x=x,
                y_pred=y_pred,
                probabilities=probs,
                committee_predictions=committee_predictions,
                committee_probabilities=None
            )
            
            if uses_cluster_hardness: 
                block_scores.append(strategy.last_score)

            if queried:
                # salva a posição exata da query no stream
                history["query_positions"].append(
                    round(100 * (i + 1) / n, 2)
                )

                history["query_indices"].append(i)

                # atualiza hardness usando o erro PRE-TREINO
                if uses_cluster_hardness:

                    strategy.update_after_query(
                        x=x,
                        y_true=y,
                        #y_pred=y_pred
                    )

                model.train(x, y)

                if committee is not None:
                    committee.train(x, y)
            
            # atualizar threshold apenas ao final do bloco
            if (
                mode == "online"
                and isinstance(strategy, (UncertaintySampling, QueryByCommittee, ClusterHardness))
                and (i + 1) % block_size == 0
            ):

                print(f"Bloco finalizado: {i+1}")

                print(f"Scores usados: {len(block_scores)}")

                print(f"Threshold antigo: {strategy.adaptive_threshold.get_threshold()}")

                strategy.adaptive_threshold.update_block(block_scores)

                print(f"Threshold novo: {strategy.adaptive_threshold.get_threshold()}")

                block_scores = []

        # POOL
        else:
            pool.add(
                x=x,
                y=y,
                y_pred=y_pred,
                probabilities=probs,
                committee_predictions=committee_predictions,
                stream_index=i
            )

            if pool.is_full():
                selected = strategy.query_pool(pool)

                for idx in selected:
                    inst = pool.instances[idx]

                    history["query_positions"].append(
                        round(100 * (inst["stream_index"] + 1) / n, 2)
                    )

                    history["query_indices"].append( inst["stream_index"] )

                    if isinstance(strategy, PoolClusterHardness):

                        #cluster_id = strategy.cluster_manager.get_cluster(inst["x"])

                        strategy.update_after_query(
                            x=inst["x"],
                            y_true=inst["y"]
                        )

                    model.train(inst["x"], inst["y"])

                    if committee is not None:
                        committee.train(inst["x"], inst["y"])

                pool.clear()

    if mode == "pool" and len(pool) > 0:
        selected = strategy.query_pool(pool)

        for idx in selected:
            inst = pool.instances[idx]

            history["query_positions"].append(
                round(100 * (inst["stream_index"] + 1) / n, 2)
            )

            history["query_indices"].append( inst["stream_index"] )

            if isinstance(strategy, PoolClusterHardness):
                strategy.update_after_query(
                    x=inst["x"],
                    y_true=inst["y"]
                )

            model.train(inst["x"], inst["y"])

            if committee is not None:
                committee.train(inst["x"], inst["y"])

    if (
        mode == "online"
        and isinstance(strategy, (UncertaintySampling, QueryByCommittee, ClusterHardness))
        and len(block_scores) > 0
    ):
        strategy.adaptive_threshold.update_block(block_scores)

    # tempo final
    end_time = time.time()
    execution_time = end_time - start_time

    # memória final
    memory_after = process.memory_info().rss / 1024**2

    memory_usage = max(
        0,
        memory_after - memory_before
    )

    # consultas por segundo
    queries_per_second = (
        strategy.total_queried / execution_time
        if execution_time > 0
        else 0
    )

    # proporção de instâncias consultadas
    query_rate = (
        strategy.total_queried / strategy.total_seen
        if strategy.total_seen > 0
        else 0
    )

    # resultados finais
    results = {
        "Hamming Loss": np.mean(hamming_scores),
        "Exact Match": np.mean(exact_match_scores),
        "F1-score": np.mean(f1_scores),
        "Execution Time (s)": execution_time,
        "Memory Usage (MB)": memory_usage,
        "Queried Instances": strategy.total_queried,
        "Queries per Second": queries_per_second,
        "Query Rate": query_rate,
        "Drift Points": drift_points,
        "Number of Drifts": len(drift_points),
        "History": history
    }

    print(results)

    return results

"""### Rodando os 5 datasets"""

all_results_online = {}

budgets = [0.30]#[0.10, 0.20, 0.30, 0.40, 0.50]

block_size = 20

cluster_modes = ["global", "per_label"]

k_values = {
    "birds": [5],
    "emotions": [7],
    "flags": [5],
    "scene": [6],
    "SynHPGrad": [5],
    "SynHPInc": [10],
    "water-quality": [8],
    "yeast": [7]
}

logging.info("===== START ONLINE EXPERIMENT =====")

for dataset_name in dataset_paths.keys():

    all_results_online[dataset_name] = {}

    # Estratégias sem budget
    fixed_strategies = {
        "NoQuery": NoQuery(),
        "FullSupervision": FullSupervision()
    }

    for strategy_name, strategy in fixed_strategies.items():

        print(f"\nDataset: {dataset_name}")
        print(f"Strategy: {strategy_name}")
        
        logging.info(f"START | Dataset={dataset_name} | Strategy={strategy_name} | Budget=None")

        try:
            results = run_experiment(
                dataset_name,
                strategy,
                model_class=AdaptiveRandomForestClassifier,
                mode="online"
            )

            all_results_online[dataset_name][strategy_name] = results

            save_result(
                dataset=dataset_name,
                strategy=strategy_name,
                budget=None,
                mode="online",
                results=results
            )

            save_history(
                history=results["History"],
                dataset_name=dataset_name,
                strategy_name=strategy_name,
                mode="online",
                output_dir=RESULTS_DIR,
                budget=None
            )

            logging.info(
                f"FINISHED | Dataset={dataset_name} | "
                f"Strategy={strategy_name} | Budget=None"
            )
        
        except Exception as e:
                print(f"Erro em {dataset_name} - {strategy_name} - sem budget")
                print(e)

                error_logger.exception(
                    f"ERROR | Dataset={dataset_name} | "
                    f"Strategy={strategy_name} | Budget=None"
                )


    # Estratégias com budget
    for budget in budgets:

        strategies = {
            "Random": RandomSampling(budget=budget),

            "Uncertainty": UncertaintySampling(
                adaptive_threshold=AdaptiveThreshold(block_size=block_size, budget=budget),
                budget=budget,
            ),
            
            "QBC": QueryByCommittee(
                adaptive_threshold=AdaptiveThreshold(block_size=block_size, budget=budget),
                budget=budget,
                warmup=0
            )
        }

        for strategy_name, strategy in strategies.items():

            print(f"\nDataset: {dataset_name}")
            print(f"Strategy: {strategy_name} | Budget: {int(budget*100)}%")
            
            logging.info(f"START | Dataset={dataset_name} | Strategy={strategy_name} | Budget={budget}")

            try:
                if strategy_name == "Random":

                    all_runs = []
                    query_runs = []

                    for seed in range(5):

                        strategy = RandomSampling(budget=budget)

                        results = run_experiment(
                            dataset_name,
                            strategy,
                            model_class=AdaptiveRandomForestClassifier,
                            mode="online",
                            random_seed=seed
                        )

                        all_runs.append(results)
                        query_runs.append({
                            "queries": results["History"]["queries"],
                            "query_positions": results["History"]["query_positions"],
                            "query_indices": results["History"]["query_indices"]
                        })

                    results = average_results(all_runs)
                
                if strategy_name == "Random":
                    save_random_queries(
                        query_runs=query_runs,
                        dataset_name=dataset_name,
                        budget=budget,
                        output_dir=RESULTS_DIR
                    )

                else:

                    results = run_experiment(
                        dataset_name,
                        strategy,
                        model_class=AdaptiveRandomForestClassifier,
                        mode="online"
                    )

                key = f"{strategy_name}_{int(budget*100)}%"
                all_results_online[dataset_name][key] = results

                save_result(
                    dataset=dataset_name,
                    strategy=strategy_name,
                    budget=budget,
                    mode="online",
                    results=results
                )

                save_history(
                    history=results["History"],
                    dataset_name=dataset_name,
                    strategy_name=strategy_name,
                    mode="online",
                    output_dir=RESULTS_DIR,
                    budget=budget,
                    repetitions=5 if strategy_name == "Random" else None
                )

                if strategy_name == "Random":
                    save_random_queries(
                        query_runs=query_runs,
                        dataset_name=dataset_name,
                        budget=budget,
                        output_dir=RESULTS_DIR
                    )

                logging.info(
                    f"FINISHED | Dataset={dataset_name} | "
                    f"Strategy={strategy_name} | Budget={budget}"
                )

            except Exception as e:
                print(f"Erro em {dataset_name} - {strategy_name} - {budget}")
                print(e)

                error_logger.exception(
                    f"ERROR | Dataset={dataset_name} | "
                    f"Strategy={strategy_name} | Budget={budget}"
                )

        # ClusterHardness: testa os diferentes valores de K e modos
        for cluster_mode in cluster_modes:

            for k in k_values.get(dataset_name, [10]):

                strategies = {
                    "ClusterHardness": ClusterHardness(
                        adaptive_threshold=AdaptiveThreshold(
                            block_size=block_size,
                            budget=budget
                        ),
                        budget=budget,
                        cluster_manager=ClusterManager(
                            n_clusters=k,
                            random_state=42,
                            mode=cluster_mode
                        ),
                        alpha=0.05
                    ),

                    "ClusterHardnessUncertainty": ClusterHardnessUncertainty(
                        adaptive_threshold=AdaptiveThreshold(
                            block_size=block_size,
                            budget=budget
                        ),
                        budget=budget,
                        cluster_manager=ClusterManager(
                            n_clusters=k,
                            random_state=42,
                            mode=cluster_mode
                        ),
                        alpha=0.05,
                        uncertainty_weight=0.5
                    )
                }

            for strategy_name, strategy in strategies.items():

                print(f"\nDataset: {dataset_name}")
                print(
                    f"Strategy: {strategy_name} | "
                    f"Mode: {cluster_mode} | "
                    f"K: {k} | "
                    f"Budget: {int(budget*100)}%"
                )

                logging.info(
                    f"START | Dataset={dataset_name} | "
                    f"Strategy={strategy_name} | "
                    f"Mode={cluster_mode} | "
                    f"K={k} | Budget={budget}"
                )

                try:

                    results = run_experiment(
                        dataset_name,
                        strategy,
                        model_class=AdaptiveRandomForestClassifier,
                        mode="online"
                    )

                    key = (
                        f"{strategy_name}_{cluster_mode}_K{k}_"
                        f"{int(budget*100)}%"
                    )

                    all_results_online[dataset_name][key] = results

                    save_result(
                        dataset=dataset_name,
                        strategy=f"{strategy_name}_{cluster_mode}_K{k}",
                        budget=budget,
                        mode="online",
                        results=results
                    )

                    save_history(
                        history=results["History"],
                        dataset_name=dataset_name,
                        strategy_name=f"{strategy_name}_{cluster_mode}_K{k}",
                        mode="online",
                        output_dir=RESULTS_DIR,
                        budget=budget
                    )

                    logging.info(
                        f"FINISHED | Dataset={dataset_name} | "
                        f"Strategy={strategy_name} | "
                        f"Mode={cluster_mode} | "
                        f"K={k} | Budget={budget}"
                    )

                except Exception as e:

                    print(
                        f"Erro em {dataset_name} - "
                        f"{strategy_name} - Mode={cluster_mode} - K={k} - {budget}"
                    )
                    print(e)

                    error_logger.exception(
                        f"ERROR | Dataset={dataset_name} | "
                        f"Strategy={strategy_name} | "
                        f"Mode={cluster_mode} | "
                        f"K={k} | Budget={budget}"
                    ) 


all_results_pool = {}

budgets = [0.30]#[0.10, 0.20, 0.30, 0.40, 0.50]

pool_size = 100

logging.info("===== START POOL EXPERIMENT =====")

for dataset_name in dataset_paths.keys():

    all_results_pool[dataset_name] = {}

    for budget in budgets:

        # Estratégias que NÃO dependem de K

        strategies = {
            "PoolRandom": PoolRandomSampling(budget=budget),
            "PoolUncertainty": PoolUncertaintySampling(budget=budget),
            "PoolQBC": PoolQueryByCommittee(budget=budget)
        }

        for strategy_name, strategy in strategies.items():

            print(f"\nDataset: {dataset_name}")
            print(f"Strategy: {strategy_name} | Budget: {int(budget*100)}%")
                
            logging.info(f"START | Dataset={dataset_name} | Strategy={strategy_name} | Budget={budget}")

            try:
                if strategy_name == "PoolRandom":

                    all_runs = []

                    for seed in range(5):

                        strategy = PoolRandomSampling(budget=budget)

                        results = run_experiment(
                            dataset_name,
                            strategy,
                            model_class=AdaptiveRandomForestClassifier,
                            mode="pool",
                            pool_size=pool_size,
                            random_seed=seed
                        )

                        all_runs.append(results)

                    results = average_results(all_runs)

                else:

                    results = run_experiment(
                    dataset_name,
                    strategy,
                    model_class=AdaptiveRandomForestClassifier,
                    mode="pool",
                    pool_size=pool_size
                )

                key = f"{strategy_name}_{int(budget*100)}%"
                all_results_pool[dataset_name][key] = results

                save_result(
                    dataset=dataset_name,
                    strategy=strategy_name,
                    budget=budget,
                    mode="pool",
                    results=results
                )

                save_history(
                    history=results["History"],
                    dataset_name=dataset_name,
                    strategy_name=strategy_name,
                    mode="pool",
                    output_dir=RESULTS_DIR,
                    budget=budget,
                    repetitions=5 if strategy_name == "PoolRandom" else None
                )

                logging.info(
                    f"FINISHED | Dataset={dataset_name} | "
                    f"Strategy={strategy_name} | Budget={budget}"
                )

            except Exception as e:
                print(f"Erro em {dataset_name} - {strategy_name} - {budget}")
                print(e)
                    
                error_logger.exception(
                    f"ERROR | Dataset={dataset_name} | "
                    f"Strategy={strategy_name} | Budget={budget}"
                )
        
        # PoolClusterHardness: Testando os diferentes valores de K
        for cluster_mode in cluster_modes:

            for k in k_values.get(dataset_name, [10]):

                pool_strategies = {
                    "PoolClusterHardness": PoolClusterHardness,
                    "PoolClusterHardnessUncertainty": PoolClusterHardnessUncertainty
                }

                for strategy_name, strategy_class in pool_strategies.items():

                    print(f"\nDataset: {dataset_name}")
                    print(
                        f"Strategy: {strategy_name} | "
                        f"Mode: {cluster_mode} | "
                        f"K: {k} | "
                        f"Budget: {int(budget*100)}%"
                    )

                    logging.info(
                        f"START POOL | Dataset={dataset_name} | "
                        f"Strategy={strategy_name} | "
                        f"Mode={cluster_mode} | "
                        f"K={k} | Budget={budget}"
                    )

                    try:

                        # ClusterManager
                        cluster_manager = ClusterManager(
                            n_clusters=k,
                            random_state=42,
                            mode=cluster_mode
                        )

                        # Hardness
                        hardness = {}

                        if strategy_name == "PoolClusterHardness":

                            strategy = PoolClusterHardness(
                                budget=budget,
                                cluster_manager=cluster_manager,
                                hardness=hardness,
                                alpha=0.05
                            )

                        else:

                            strategy = PoolClusterHardnessUncertainty(
                                budget=budget,
                                cluster_manager=cluster_manager,
                                hardness=hardness,
                                alpha=0.05,
                                uncertainty_weight=0.5
                            )

                        results = run_experiment(
                            dataset_name,
                            strategy,
                            model_class=AdaptiveRandomForestClassifier,
                            mode="pool",
                            pool_size=pool_size
                        )

                        key = (
                            f"{strategy_name}_"
                            f"{cluster_mode}_"
                            f"K{k}_"
                            f"{int(budget*100)}%"
                        )

                        all_results_pool[dataset_name][key] = results

                        save_result(
                            dataset=dataset_name,
                            strategy=f"{strategy_name}_{cluster_mode}_K{k}",
                            budget=budget,
                            mode="pool",
                            results=results
                        )

                        save_history(
                            history=results["History"],
                            dataset_name=dataset_name,
                            strategy_name=f"{strategy_name}_{cluster_mode}_K{k}",
                            mode="pool",
                            output_dir=RESULTS_DIR,
                            budget=budget
                        )

                        logging.info(
                            f"FINISHED POOL | Dataset={dataset_name} | "
                            f"Strategy={strategy_name} | "
                            f"Mode={cluster_mode} | "
                            f"K={k} | Budget={budget}"
                        )

                    except Exception as e:

                        print(
                            f"Erro em {dataset_name} - "
                            f"{strategy_name} - Mode={cluster_mode} - "
                            f"K={k} - {budget}"
                        )
                        print(e)

                        error_logger.exception(
                            f"ERROR POOL | Dataset={dataset_name} | "
                            f"Strategy={strategy_name} | "
                            f"Mode={cluster_mode} | "
                            f"K={k} | Budget={budget}"
                        )

logging.info("===== ALL EXPERIMENTS FINISHED =====")
