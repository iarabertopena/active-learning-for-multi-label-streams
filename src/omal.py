### Imports
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
TFELM = ROOT / "TfELM"

if str(TFELM) not in sys.path:
    sys.path.insert(0, str(TFELM))

import numpy as np
import psutil

from sklearn.metrics import (hamming_loss, f1_score)
from sklearn.preprocessing import StandardScaler

from Models.ELMModel import ELMModel
from Layers.WELMLayer import WELMLayer

from datasets import load_multilabel_dataset

### Chunk sizes
BLOCK_SIZES = {
    "flags": 20,
    "CHD_49": 50,
    "emotions": 50,
    "medical": 100,
    "water-quality": 100,
    "image": 200,
    "scene": 200,
    "yeast": 200,
    "EukaryotePseAAC": 500,
    "yelp": 800,
}

### Função para criar um novo WELM
def create_welm(
    number_neurons=1000,
    activation="sigmoid",
    C=1.0,
    weight_method="wei-1"
):

    layer = WELMLayer(
        number_neurons=number_neurons,
        activation=activation,
        C=C,
        weight_method=weight_method,
    )

    model = ELMModel(layer)

    return model

### Classifier Chain
class ClassifierChain:

    def __init__(self, base_model_fn, label_order):

        self.base_model_fn = base_model_fn
        self.label_order = np.array(label_order)
        self.classifiers = []


    def fit(self, X, Y):

        self.classifiers = []

        Y_ordered = Y[:, self.label_order]

        X_chain = X.copy()


        for i in range(Y.shape[1]):

            #print(f"Training label {i+1}/{Y.shape[1]}")

            model = self.base_model_fn()

            y_label = Y_ordered[:, i].reshape(-1,1)

            model.fit(
                X_chain,
                y_label
            )

            self.classifiers.append(model)

            X_chain = np.column_stack(
                [
                    X_chain,
                    y_label#.ravel()
                ]
            )

        return self

    def predict_proba(self, X):

        X_chain = X.copy()

        predictions = []

        for clf in self.classifiers:
            prob = clf.predict_proba(X_chain)

            if hasattr(prob, "numpy"):
                prob = prob.numpy()

            prob = np.asarray(prob)

            # WELM binário no OMAL deve retornar (n_samples, 1)
            if prob.ndim == 2 and prob.shape[1] == 1:
                prob = prob[:, 0]

            # Caso algum classificador retorne duas probabilidades
            elif prob.ndim == 2 and prob.shape[1] == 2:
                prob = prob[:, 1]

            else:
                raise ValueError(
                    f"Unexpected probability shape: {prob.shape}"
                )

            predictions.append(prob)

            pred = (prob >= 0.5).astype(int)

            X_chain = np.column_stack(
                [
                    X_chain,
                    pred
                ]
            )

        predictions = np.column_stack(predictions)

        inverse_order = np.argsort(self.label_order)

        predictions = predictions[:, inverse_order]

        return predictions

    def predict(self, X):

        prob = self.predict_proba(X)

        return (prob >= 0.5).astype(int)

### OMAL
class OMAL:

    def __init__(
        self,
        number_neurons=1000,
        activation="sigmoid",
        C=1.0,
        weight_method="wei-1",
        lambda_=0.5,
        query_rate=0.6,
    ):

        self.number_neurons = number_neurons
        self.activation = activation
        self.C = C
        self.weight_method = weight_method

        self.lambda_ = lambda_
        self.query_rate = query_rate

        self.classifier = None
        self.label_order = None
      
        self.history = {
            "progress": [],
            "f1_macro": [],
            "f1_micro": [],
            "hamming": [],
            "exact_match": [],
            "queries": [],
            "evaluated": []
        }

    def compute_label_ranking(self, Y):
        """
        Compute the global label ranking sequence R (Eq. 6 of OMAL).

        Parameters
        ----------
        Y : ndarray of shape (n_samples, n_labels)
            Binary label matrix.

        Returns
        -------
        ranking : ndarray of shape (n_labels,)
            Label indices ordered from most to least significant.
        """

        n_labels = Y.shape[1]

        # Step 1 - Co-occurrence matrix
        cooccurrence = Y.T @ Y
        cooccurrence = cooccurrence.astype(float)

        # Ignore self-co-occurrence
        np.fill_diagonal(cooccurrence, 0)

        # Step 2 - Local ranking matrix
        local_rank = np.zeros((n_labels, n_labels), dtype=int)

        for row in range(n_labels):

            # Labels ordered by decreasing co-occurrence
            order = np.argsort(-cooccurrence[row])

            # Convert ordering into ranks (1 = highest)
            for rank, label in enumerate(order, start=1):
                local_rank[row, label] = rank

        # Step 3 - Global ranking score (Eq. 6)
        scores = np.zeros(n_labels)

        for i in range(n_labels):

            denominator = 0.0

            for j in range(n_labels):

                if i == j:
                    continue

                denominator += 1.0 / local_rank[j, i]

            scores[i] = 1.0 / denominator

        # Step 4 - Global ranking sequence
        ranking = np.argsort(scores)[::-1]

        return ranking

    def _build_classifier(self):

        classifier = ClassifierChain(
            base_model_fn=lambda: create_welm(
                number_neurons=self.number_neurons,
                activation=self.activation,
                C=self.C,
                weight_method=self.weight_method,
            ),
            label_order=self.label_order
        )

        return classifier

    def fit(self, X, Y):

        print("Training:", X.shape)

        self.label_order = self.compute_label_ranking(Y)

        #self.label_order = np.arange(Y.shape[1])

        print("Label order:", self.label_order)
        print("Label distribution:")
        print(Y.mean(axis=0))

        self.classifier = self._build_classifier()

        self.classifier.fit(X, Y)

        return self

    def _predict_chain(self, X):

        probabilities = self.classifier.predict_proba(X)

        predictions = (probabilities >= 0.5).astype(int)

        return predictions, probabilities

    def predict(self, X):

        predictions, _ = self._predict_chain(X)

        return predictions


    def predict_proba(self, X):

        _, probabilities = self._predict_chain(X)

        return probabilities
    
    def compute_entropy(self, probabilities):
        """
        Compute the average label entropy (Eq. 2 of OMAL).

        Parameters
        ----------
        probabilities : ndarray of shape (n_samples, n_labels)
            Predicted probabilities P(y=1) for each label.

        Returns
        -------
        entropy : ndarray of shape (n_samples,)
            Average entropy of each instance.
        """

        # Avoid log(0)
        eps = 1e-12
        probabilities = np.clip(probabilities, eps, 1.0 - eps)

        entropy = -np.mean(
            probabilities * np.log(probabilities)
            + (1.0 - probabilities) * np.log(1.0 - probabilities),
            axis=1,
        )

        return entropy
    
    def compute_diversity(self, X_query, X_labeled):
        """
        Compute the diversity score (Eq. 4 of OMAL).

        Parameters
        ----------
        X_query : ndarray of shape (n_query, n_features)
            Candidate instances.

        X_labeled : ndarray of shape (n_labeled, n_features)
            Labeled instances.

        Returns
        -------
        diversity : ndarray of shape (n_query,)
            Diversity score of each candidate instance.
        """

        # L2 norms
        norm_query = np.linalg.norm(X_query, axis=1, keepdims=True)
        norm_labeled = np.linalg.norm(X_labeled, axis=1, keepdims=True)

        # Avoid division by zero
        eps = 1e-12
        norm_query = np.clip(norm_query, eps, None)
        norm_labeled = np.clip(norm_labeled, eps, None)

        # Eq. (3): cosine similarity
        similarity = (X_query @ X_labeled.T) / (norm_query * norm_labeled.T)

        # Eq. (4): Diversity
        diversity = 1.0 - np.mean(similarity, axis=1)

        return diversity

    def compute_score(self, entropy, diversity):
        """
        Compute the OMAL selection score (Eq. 5).

        Parameters
        ----------
        entropy : ndarray of shape (n_samples,)
            Entropy values (Eq. 2).

        diversity : ndarray of shape (n_samples,)
            Diversity values (Eq. 4).

        Returns
        -------
        score : ndarray of shape (n_samples,)
            OMAL score for each candidate instance.
        """

        score = (
            self.lambda_ * entropy
            + (1.0 - self.lambda_) * diversity
        )

        return score
    
    def query(self, X_query, X_labeled):
        """
        Select the most significant instances according to OMAL (Algorithm 1,
        Steps 9–13).

        Parameters
        ----------
        X_query : ndarray of shape (n_samples, n_features)
            Incoming unlabeled data chunk.

        X_labeled : ndarray of shape (n_labeled, n_features)
            Current labeled set.

        Returns
        -------
        selected : ndarray of shape (n_selected,)
            Indices of the selected instances.

        scores : ndarray of shape (n_samples,)
            Significance score of every instance.
        """

        # Step 9 (Eq. 1)
        probabilities = self.predict_proba(X_query)

        # Step 10 (Eq. 2) -> Uncertainty Information
        entropy = self.compute_entropy(probabilities)

        # Step 11 (Eq. 3–4) -> Diversity
        diversity = self.compute_diversity(X_query, X_labeled)

        # Step 12 (Eq. 5)
        scores = self.compute_score(entropy, diversity)

        # Step 13
        n_selected = max(1, int(np.ceil(self.query_rate * len(X_query))))

        selected = np.argsort(scores)[::-1][:n_selected]

        return selected, scores
    
    def update_history(
        self,
        progress,
        y_true,
        y_pred,
        queries,
        evaluated
    ):

        self.history["progress"].append(progress)

        self.history["hamming"].append(
            hamming_loss(y_true, y_pred)
        )

        self.history["exact_match"].append(
            np.mean(
                np.all(y_true == y_pred, axis=1)
            )
        )

        self.history["f1_macro"].append(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        )

        self.history["f1_micro"].append(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            )
        )
        
        self.history["queries"].append(queries)

        self.history["evaluated"].append(evaluated)

# Loop principal
def run_experiment_omal(dataset_name, random_seed=None):

    print(f"Dataset: {dataset_name}")

    if random_seed is not None:
        np.random.seed(random_seed)

    # Load dataset
    X, Y = load_multilabel_dataset(dataset_name)

    # Randomly transform dataset into a stream (following the paper)
    permutation = np.random.permutation(len(X))
    X = X[permutation]
    Y = Y[permutation]

    # Split: 70% stream, 30% fixed test set
    split = int(0.70 * len(X))

    X_stream, Y_stream = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]

    # Normalize features
    scaler = StandardScaler()

    X_stream = scaler.fit_transform(X_stream)
    X_test = scaler.transform(X_test)

    print(f"Stream: {len(X_stream)}")
    print(f"Test: {len(X_test)}")

    stream_size = len(X_stream)
    total_queried = 0

    # Block size
    chunk_size = BLOCK_SIZES[dataset_name]
    print(f"Block size: {chunk_size}")

    # Initial labeled set ΦL (first block u1)
    initial_size = chunk_size

    X_labeled = X_stream[:initial_size].copy()
    Y_labeled = Y_stream[:initial_size].copy()

    X_stream = X_stream[initial_size:]
    Y_stream = Y_stream[initial_size:]

    # Initial OMAL model
    model = OMAL(query_rate=0.60, lambda_=0.5)
    model.fit(X_labeled, Y_labeled)

    start_time = time.time()

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024**2

    # Initial point of learning curve (u1)
    Y_test_pred = model.predict(X_test)

    model.history["progress"].append(0.0)
    model.history["f1_macro"].append(
        f1_score(Y_test, Y_test_pred, average="macro", zero_division=0)
    )
    model.history["f1_micro"].append(
        f1_score(Y_test, Y_test_pred, average="micro", zero_division=0)
    )
    model.history["hamming"].append(
        hamming_loss(Y_test, Y_test_pred)
    )
    model.history["exact_match"].append(
        np.mean(np.all(Y_test == Y_test_pred, axis=1))
    )
    model.history["queries"].append(0)
    model.history["evaluated"].append(len(X_labeled))

    # Stream processing (Algorithm 1)
    processed_instances = 0

    for start in range(0, len(X_stream), chunk_size):

        end = min(start + chunk_size, len(X_stream))

        X_chunk = X_stream[start:end]
        Y_chunk = Y_stream[start:end]

        # OMAL query (Steps 8-13)
        selected, _ = model.query(X_query=X_chunk, X_labeled=X_labeled)

        total_queried += len(selected)

        # Oracle (Step 14)
        X_selected = X_chunk[selected]
        Y_selected = Y_chunk[selected]

        # Update labeled set ΦL (Step 15)
        X_labeled = np.vstack([X_labeled, X_selected])
        Y_labeled = np.vstack([Y_labeled, Y_selected])

        # Retrain OMAL (Steps 17-20)
        model.fit(X_labeled, Y_labeled)

        # Evaluate on fixed test set
        Y_test_pred = model.predict(X_test)

        f1_macro = f1_score(Y_test, Y_test_pred, average="macro", zero_division=0)

        f1_micro = f1_score(Y_test, Y_test_pred, average="micro", zero_division=0)

        hamming = hamming_loss(Y_test, Y_test_pred)

        exact = np.mean(np.all(Y_test == Y_test_pred, axis=1))

        processed_instances += len(X_chunk)
        progress = processed_instances / len(X_stream)

        model.history["progress"].append(progress)
        model.history["f1_macro"].append(f1_macro)
        model.history["f1_micro"].append(f1_micro)
        model.history["hamming"].append(hamming)
        model.history["exact_match"].append(exact)
        model.history["queries"].append(len(selected))
        model.history["evaluated"].append(len(X_labeled))

    # Final evaluation on fixed test set
    Y_pred = model.predict(X_test)

    test_hamming = hamming_loss(Y_test, Y_pred)

    test_exact = np.mean(
        np.all(Y_test == Y_pred, axis=1)
    )

    test_f1_macro = f1_score(
        Y_test,
        Y_pred,
        average="macro",
        zero_division=0
    )

    test_f1_micro = f1_score(
        Y_test,
        Y_pred,
        average="micro",
        zero_division=0
    )

    end_time = time.time()
    memory_after = process.memory_info().rss / 1024**2

    results = {
        "Hamming Loss": test_hamming,
        "Exact Match": test_exact,
        "F1 Macro": test_f1_macro,
        "F1 Micro": test_f1_micro,
        "Execution Time (s)": end_time - start_time,
        "Memory Usage (MB)": max(0, memory_after - memory_before),
        "Queried Instances": total_queried,
        "Query Rate": total_queried / stream_size,
        "History": model.history
    }

    print(results)

    return results

# TESTE
if __name__ == "__main__":

    #for dataset in BLOCK_SIZES:

    #run_experiment_omal(dataset_name=dataset, random_seed=0)
    run_experiment_omal(dataset_name="emotions", random_seed=0)
