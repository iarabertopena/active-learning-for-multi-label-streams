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
from sklearn.cluster import MiniBatchKMeans

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
    #"water-quality": DATASET_DIR / "water-quality.arff",
    #"SynHPGrad": DATASET_DIR / "SynHPGrad.arff",
    #"SynHPInc": DATASET_DIR / "SynHPInc.arff",
}

n_labels_dict = {
    "flags": 7,
    "emotions": 6,
    "scene": 6,
    "birds": 19,
    "yeast": 14,
    #"water-quality": 14,
    #"SynHPGrad": 8,
    #"SynHPInc": 8
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

        print(
            f"Seen={self.total_seen} | "
            f"Threshold={threshold:.4f} | "
            f"Uncertainty={uncertainty:.4f}"
        )
        
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

        print(
            f"Seen={self.total_seen} | "
            f"Threshold={threshold:.10f} | "
            f"Disagreement={disagreement:.10f} | "
            f"Query={disagreement >= threshold}"
        )

        if disagreement >= threshold:
            self.total_queried += 1

            return True

        return False

"""#### Cluster-Hardness Sampling (CHS)"""

class ClusterManager:

    def __init__(
        self,
        n_clusters=10,
        random_state=42,
        batch_size=100
    ):

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.batch_size = batch_size

        self.model = None

    def fit(self, X):

        n_clusters = min(self.n_clusters, len(X))

        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            batch_size=min(self.batch_size, len(X)),
            n_init=10
        )

        self.model.fit(X)

    def get_cluster(self, x):

        x = np.asarray(x).reshape(1, -1)

        return int(self.model.predict(x)[0])

    def update(self, x):

        x = np.asarray(x).reshape(1, -1)

        self.model.partial_fit(x)

    @property
    def n_clusters_actual(self):

        if self.model is None:
            return 0

        return self.model.n_clusters

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

        # H_c
        self.hardness = {}

        # número de observações usadas para cada cluster
        self.cluster_counts = {}
        
        self.last_cluster_id = None

    def initialize_hardness(
        self,
        x,
        difficulty
    ):

        cluster_id = self.cluster_manager.get_cluster(x)

        if cluster_id not in self.hardness:

            self.hardness[cluster_id] = difficulty
            self.cluster_counts[cluster_id] = 1

        else:

            self.update_hardness(cluster_id, difficulty)

    def update_hardness(
        self,
        cluster_id,
        difficulty
    ):

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

    def get_hardness(self, cluster_id):

        # cluster ainda não observado
        if cluster_id not in self.hardness:

            return 0.0

        return self.hardness[cluster_id]

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
        cluster_id = self.cluster_manager.get_cluster(x)
        # obtém a hardness histórica
        hardness = self.get_hardness(cluster_id)
        # threshold atual
        threshold = self.adaptive_threshold.get_threshold()

        print(
            f"Seen={self.total_seen} | "
            f"Cluster={cluster_id} | "
            f"Hardness={hardness:.10f} | "
            f"Threshold={threshold:.10f}"
        )

        # guarda o cluster usado na decisão
        self.last_cluster_id = cluster_id

        # atualiza o clustering usando somente X
        self.cluster_manager.update(x)

        if hardness >= threshold:

            self.total_queried += 1

            return True

        return False

    def update_after_query(
        self,
        x,
        y_true,
        y_pred
    ):
        # usa o mesmo cluster que foi usado na decisão da query
        cluster_id = self.last_cluster_id
        # Calcula o erro da predição
        difficulty = hamming_loss(y_true, y_pred)
        #atualiza a hardness
        self.update_hardness(cluster_id, difficulty)

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
        hardness
    ):

        super().__init__(budget)

        self.cluster_manager = cluster_manager
        self.hardness = hardness

    def query_pool(self, pool):

        remaining_budget = (self.budget_limit - self.total_queried)

        if remaining_budget <= 0:
            return []

        scores = []

        for inst in pool.instances:

            cluster_id = self.cluster_manager.get_cluster(inst["x"])

            score = self.hardness.get_hardness(cluster_id)

            scores.append(score)

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

"""### Loop principal"""
def run_experiment(
    dataset_name,
    strategy,
    model_class,
    mode="online",
    pool_size=None,
    block_size=None,
    random_seed=None
):

    print(f"Dataset: {dataset_name}")

    if random_seed is not None:
        np.random.seed(random_seed)

    # carregar dados
    X, Y = load_multilabel_dataset(dataset_name)

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
        (UncertaintySampling, PoolUncertaintySampling)
    )

    uses_committee = isinstance(
        strategy,
        (QueryByCommittee, PoolQueryByCommittee)
    )

    uses_cluster_hardness = isinstance(
        strategy,
        ClusterHardness
    )

    # ClusterHardness -> inicializa clusters antes de aprender a instância
    if uses_cluster_hardness:
        strategy.cluster_manager.fit(X_train)

        print(
            f"Clusters inicializados: "
            f"{strategy.cluster_manager.n_clusters_actual}"
        )

    # treinamento inicial (_% do dataset)
    for i in range(len(X_train)):

        x_train = X_train[i]
        y_train = Y_train[i]

        # ClusterHardnes -> prever antes de aprender a instância
        if uses_cluster_hardness:

            if i == 0:
                difficulty = 0.5
            
            else:
                y_pred_train = model.predict(x_train)

                difficulty = hamming_loss(y_train, y_pred_train)

            strategy.initialize_hardness(x_train, difficulty)

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
    if mode == "online" and isinstance(strategy, UncertaintySampling):

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
    
    if mode == "online" and isinstance(strategy, ClusterHardness):#uses_cluster_hardness:

        scores = list(strategy.hardness.values())

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

            if mode == "online":
                score = compute_uncertainty(probs)
                block_scores.append(score)

        # previsões do comitê
        committee_predictions = None

        if uses_committee:
            committee_predictions = committee.predict_all(x)

            if mode == "online":
                score = compute_vote_entropy(committee_predictions)
                block_scores.append(score)

        
        # ClusterHardness -> inicializa clusters antes de aprender a instância
        if uses_cluster_hardness:
            cluster_id = strategy.cluster_manager.get_cluster(x)
            score = strategy.hardness.get(cluster_id, 0.5)

            if mode == "online":
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
                        y_pred=y_pred
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

                        cluster_id = strategy.cluster_manager.get_cluster(inst["x"])

                        difficulty = hamming_loss(inst["y"], inst["y_pred"])

                        strategy.update_hardness(cluster_id, difficulty)

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
            ),

            "ClusterHardness_0.05": ClusterHardness(
                adaptive_threshold=AdaptiveThreshold(
                    block_size=block_size,
                    budget=budget
                ),
                budget=budget,
                cluster_manager=ClusterManager(
                    n_clusters=10,
                    random_state=42
                ),
                alpha=0.05
            ),
            "ClusterHardness_0.1": ClusterHardness(
                adaptive_threshold=AdaptiveThreshold(
                    block_size=block_size,
                    budget=budget
                ),
                budget=budget,
                cluster_manager=ClusterManager(
                    n_clusters=10,
                    random_state=42
                ),
                alpha=0.1
            ),
            "ClusterHardness_0.3": ClusterHardness(
                adaptive_threshold=AdaptiveThreshold(
                    block_size=block_size,
                    budget=budget
                ),
                budget=budget,
                cluster_manager=ClusterManager(
                    n_clusters=10,
                    random_state=42
                ),
                alpha=0.3
            ),
            "ClusterHardness_0.5": ClusterHardness(
                adaptive_threshold=AdaptiveThreshold(
                    block_size=block_size,
                    budget=budget
                ),
                budget=budget,
                cluster_manager=ClusterManager(
                    n_clusters=10,
                    random_state=42
                ),
                alpha=0.5
            ),
            "ClusterHardness_0.7": ClusterHardness(
                adaptive_threshold=AdaptiveThreshold(
                    block_size=block_size,
                    budget=budget
                ),
                budget=budget,
                cluster_manager=ClusterManager(
                    n_clusters=10,
                    random_state=42
                ),
                alpha=0.7
            ),
            "ClusterHardness_0.9": ClusterHardness(
                adaptive_threshold=AdaptiveThreshold(
                    block_size=block_size,
                    budget=budget
                ),
                budget=budget,
                cluster_manager=ClusterManager(
                    n_clusters=10,
                    random_state=42
                ),
                alpha=0.9
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

all_results_pool = {}

budgets = [0.30]#[0.10, 0.20, 0.30, 0.40, 0.50]

pool_size = 100

logging.info("===== START POOL EXPERIMENT =====")

for dataset_name in dataset_paths.keys():

    all_results_pool[dataset_name] = {}

    for budget in budgets:

        strategies = {
            "PoolRandom": PoolRandomSampling(budget=budget),
            "PoolUncertainty": PoolUncertaintySampling(budget=budget),
            "PoolQBC": PoolQueryByCommittee(budget=budget),
            "PoolClusterHardness": PoolClusterHardness(
                budget=budget,
                cluster_manager=cluster_manager,
                hardness=hardness
            )
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

logging.info("===== ALL EXPERIMENTS FINISHED =====")