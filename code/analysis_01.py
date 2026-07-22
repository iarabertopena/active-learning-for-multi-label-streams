import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json

PROJECT_DIR = Path("/disk2/iara/projetos/active_learning_multilabel")

RUN_FOLDER = PROJECT_DIR / "results" / "run_20260716_124512"

FIGURES_DIR = RUN_FOLDER / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

df_results_online = pd.read_csv(RUN_FOLDER / "online_results.csv")
print(df_results_online.head())

df_results_pool = pd.read_csv(RUN_FOLDER / "pool_results.csv")
print(df_results_pool.head())

def save_figure(filename, output_dir=RUN_FOLDER):

    # Salva a figura atual do matplotlib dentro da pasta figures.

    plt.savefig(
        figures_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

"""### Análise dos resultados"""

strategy_names = {
    "NoQuery": "Lower Bound",
    "FullSupervision": "Upper Bound",
    "Random": "Random Sampling",
    "Uncertainty": "Uncertainty Sampling",
    "QBC": "Query-by-Committee",
    "PoolRandom": "Pool Random Sampling",
    "PoolUncertainty": "Pool Uncertainty Sampling",
    "PoolQBC": "Pool Query-by-Committee"
}

def plot_metric(df, metric, mode):

    strategies = df["Strategy"].dropna().unique()

    for strategy in strategies:

        # Ignora estratégias sem budget
        if strategy in ["NoQuery", "FullSupervision"]:
            continue

        df_plot = df[df["Strategy"] == strategy]

        pivot = df_plot.pivot(
            index="Dataset",
            columns="Budget",
            values=metric
        )

        pivot = pivot.sort_index()

        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        pivot.columns = [f"{int(b*100)}%" for b in pivot.columns]

        ax = pivot.plot(
            kind="bar",
            figsize=(9,5),
            width=0.8
        )

        ax.set_title(f"{strategy_names.get(strategy, strategy)} - {metric}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric)

        ax.grid(axis="y", alpha=0.3)

        plt.xticks(rotation=0)
        plt.legend(title="Budget")
        plt.tight_layout()

        save_figure(
            f"{strategy}_{metric.lower().replace(' ', '_')}_{mode}.png"
        )

        plt.close()

def plot_metric_history(history_dir, metric, mode):

    metric_map = {
        "Hamming Loss": "hamming",
        "Exact Match": "exact_match",
        "F1-score": "f1"
    }

    metric_key = metric_map[metric]

    histories = {}

    for history_file in history_dir.glob(f"*_{mode}.json"):

        with open(history_file, "r") as f:
            history = json.load(f)

        dataset = history["dataset"]
        strategy = history["strategy"]
        budget = history["budget"]

        histories.setdefault(dataset, {})
        histories[dataset].setdefault(budget, {})
        histories[dataset][budget][strategy] = history

    # Gera um gráfico para cada dataset e budget
    for dataset in sorted(histories.keys()):

        for budget in sorted(histories[dataset].keys()):

            plt.figure(figsize=(9, 5))

            for strategy, history in histories[dataset][budget].items():

                # Ignora estratégias sem budget
                if strategy in ["NoQuery", "FullSupervision"]:
                    continue

                plt.plot(
                    history["progress"],
                    history[metric_key],
                    linewidth=2,
                    label=strategy_names.get(strategy, strategy)
                )

            plt.title(
                f"{dataset} - {metric} ({int(budget*100)}% Budget)"
            )

            plt.xlabel("Stream Progress (%)")
            plt.ylabel(metric)

            plt.grid(axis="y", alpha=0.3)

            plt.legend(
                title="Strategy",
                loc="best"
            )

            plt.tight_layout()

            save_figure(
                f"{dataset}_{metric_key}_{int(budget*100)}_{mode}.png"
            )

            plt.close()

"""##### Histórico"""
history_dir = RUN_FOLDER / "history"

# F1-score
plot_metric_history(history_dir, "F1-score", "online")
plot_metric_history(history_dir, "F1-score", "pool")

# Hamming Loss
plot_metric_history(history_dir, "Hamming Loss", "online")
plot_metric_history(history_dir, "Hamming Loss", "pool")

# Exact Match
plot_metric_history(history_dir, "Exact Match", "online")
plot_metric_history(history_dir, "Exact Match", "pool")


"""##### Barras"""
# Hamming Loss
plot_metric(df_results_online, "Hamming Loss", "online")
plot_metric(df_results_pool, "Hamming Loss", "pool")

# Exact Match
plot_metric(df_results_online, "Exact Match", "online")
plot_metric(df_results_pool, "Exact Match", "pool")

# F1-score
plot_metric(df_results_online, "F1-score", "online")
plot_metric(df_results_pool, "F1-score", "pool")

# Tempo
plot_metric(df_results_online, "Execution Time (s)", "online")
plot_metric(df_results_pool, "Execution Time (s)", "pool")

# Memória
plot_metric(df_results_online, "Memory Usage (MB)", "online")
plot_metric(df_results_pool, "Memory Usage (MB)", "pool")
