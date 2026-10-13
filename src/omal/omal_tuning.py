from pathlib import Path
import sys
import json
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Paths
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_multilabel_dataset
from omal import ClassifierChain, create_welm


# Configurações do Grid Search
K_VALUES = list(range(100, 2001, 100))

C_VALUES = [
    1e-3,
    1e-2,
    1e-1,
    1,
    10,
    100,
    1000,
    10000,
    100000,
]

N_FOLDS = 5

RANDOM_SEED = 0


def create_classifier_chain(K, C, label_order):

    model = ClassifierChain(
        base_model_fn=lambda: create_welm(
            number_neurons=K,
            activation="sigmoid",
            C=C,
            weight_method="wei-1",
        ),
        label_order=label_order,
    )

    return model


# Label ranking
def compute_label_ranking(Y):

    n_labels = Y.shape[1]

    # Co-occurrence matrix
    cooccurrence = Y.T @ Y
    cooccurrence = cooccurrence.astype(float)

    np.fill_diagonal(cooccurrence, 0)

    # Local ranking
    local_rank = np.zeros(
        (n_labels, n_labels),
        dtype=int
    )

    for row in range(n_labels):

        order = np.argsort(
            -cooccurrence[row]
        )

        for rank, label in enumerate(
            order,
            start=1
        ):
            local_rank[row, label] = rank

    # Global ranking
    scores = np.zeros(n_labels)

    for i in range(n_labels):

        denominator = 0.0

        for j in range(n_labels):

            if i == j:
                continue

            denominator += (
                1.0 / local_rank[j, i]
            )

        scores[i] = 1.0 / denominator

    ranking = np.argsort(scores)[::-1]

    return ranking


# Avaliação de uma combinação K,C
def evaluate_parameters(
    X,
    Y,
    K,
    C,
    n_folds=5,
    random_seed=0,
):

    kfold = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_seed,
    )

    macro_scores = []
    micro_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(X),
        start=1
    ):

        X_train = X[train_idx]
        Y_train = Y[train_idx]

        X_val = X[val_idx]
        Y_val = Y[val_idx]

        # Ranking das labels usando apenas o fold de treino
        label_order = compute_label_ranking(
            Y_train
        )

        # Classifier Chain + WELM
        model = create_classifier_chain(
            K=K,
            C=C,
            label_order=label_order,
        )

        model.fit(
            X_train,
            Y_train
        )

        # Predição
        Y_pred = model.predict(
            X_val
        )

        print("DEBUG Y_true:")
        print("shape:", Y_val.shape)
        print("dtype:", Y_val.dtype)
        print("unique:", np.unique(Y_val))

        print("DEBUG Y_pred:")
        print("shape:", Y_pred.shape)
        print("dtype:", Y_pred.dtype)
        print("unique:", np.unique(Y_pred))

        # Métricas
        macro = f1_score(
            Y_val,
            Y_pred,
            average="macro",
            zero_division=0,
        )

        micro = f1_score(
            Y_val,
            Y_pred,
            average="micro",
            zero_division=0,
        )

        macro_scores.append(macro)
        micro_scores.append(micro)

        print(
            f"    Fold {fold}/{n_folds} | "
            f"Macro-F1={macro:.4f} | "
            f"Micro-F1={micro:.4f}"
        )

    return {
        "macro_f1_mean": np.mean(macro_scores),
        "macro_f1_std": np.std(macro_scores),

        "micro_f1_mean": np.mean(micro_scores),
        "micro_f1_std": np.std(micro_scores),
    }


# Grid Search
def grid_search_omal(
    dataset_name,
    random_seed=0,
):

    print(f"OMAL GRID SEARCH — {dataset_name}")

    # Load dataset
    X, Y = load_multilabel_dataset(
        dataset_name
    )

    # Randomização
    rng = np.random.default_rng(
        random_seed
    )

    permutation = rng.permutation(
        len(X)
    )

    X = X[permutation]
    Y = Y[permutation]

    # 70% stream / 30% test
    split = int(
        0.70 * len(X)
    )

    X_stream = X[:split]
    Y_stream = Y[:split]

    print(
        f"Stream: {len(X_stream)}"
    )

    print(
        f"Test: {len(X) - len(X_stream)}"
    )

    # Standardization
    scaler = StandardScaler()

    X_stream = scaler.fit_transform(
        X_stream
    )

    # Resultados
    results = []

    total_combinations = (
        len(K_VALUES) *
        len(C_VALUES)
    )

    combination = 0

    start_time = time.time()

    # Grid
    for K in K_VALUES:

        for C in C_VALUES:

            combination += 1

            print()
            print(
                f"[{combination}/{total_combinations}] "
                f"K={K}, C={C}"
            )

            scores = evaluate_parameters(
                X=X_stream,
                Y=Y_stream,
                K=K,
                C=C,
                n_folds=N_FOLDS,
                random_seed=random_seed,
            )

            row = {
                "dataset": dataset_name,
                "K": K,
                "C": C,
                **scores,
            }

            results.append(row)

            print(
                f"    Mean Macro-F1: "
                f"{scores['macro_f1_mean']:.4f}"
            )

            print(
                f"    Mean Micro-F1: "
                f"{scores['micro_f1_mean']:.4f}"
            )

    # DataFrame
    results_df = pd.DataFrame(
        results
    )

    # Melhor combinação
    best_macro = results_df.loc[
        results_df["macro_f1_mean"].idxmax()
    ]

    best_micro = results_df.loc[
        results_df["micro_f1_mean"].idxmax()
    ]

    # Diretório de resultados
    output_dir = (
        ROOT
        / "results"
        / "omal_tuning"
        / dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Salvar grid completo
    csv_path = (
        output_dir
        / "grid_search.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    # Salvar melhores parâmetros
    best_params = {

        "dataset": dataset_name,

        "macro_f1": {
            "K": int(best_macro["K"]),
            "C": float(best_macro["C"]),
            "cv_score": float(
                best_macro["macro_f1_mean"]
            ),
        },

        "micro_f1": {
            "K": int(best_micro["K"]),
            "C": float(best_micro["C"]),
            "cv_score": float(
                best_micro["micro_f1_mean"]
            ),
        },
    }

    json_path = (
        output_dir
        / "best_parameters.json"
    )

    with open(
        json_path,
        "w"
    ) as f:

        json.dump(
            best_params,
            f,
            indent=4
        )

    elapsed = (
        time.time()
        - start_time
    )

    # Resultado final
    print("GRID SEARCH FINALIZADO")
    print()

    print(f"Melhor Macro-F1:")
    print(f"    K = {best_params['macro_f1']['K']}")
    print(f"    C = {best_params['macro_f1']['C']}")
    print(
        f"    CV = "
        f"{best_params['macro_f1']['cv_score']:.4f}"
    )

    print()

    print(f"Melhor Micro-F1:")
    print(f"    K = {best_params['micro_f1']['K']}")
    print(f"    C = {best_params['micro_f1']['C']}")
    print(
        f"    CV = "
        f"{best_params['micro_f1']['cv_score']:.4f}"
    )

    print()

    print(f"Resultados salvos em:")
    print(output_dir)
    print(f"Tempo total: " f"{elapsed:.2f} s")

    return results_df, best_params


# TESTE
if __name__ == "__main__":

    DATASETS = [
        #"flags",
        #"emotions",
        #"scene",
        #"birds",
        #"yeast",
        #"water-quality",
        "SynHPGrad",
        "SynHPInc"
    ]

    for dataset in DATASETS:
        print(f"INICIANDO GRID SEARCH: {dataset}")

        grid_search_omal(dataset, random_seed=0)