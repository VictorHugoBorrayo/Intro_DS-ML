import numpy as np
import matplotlib.pyplot as plt


def plot_real_vs_pred(
    y_test,
    y_pred,
    sample_limit=None,
    sort_by_real=False,
    title="Valores reales vs predicción",
    xlabel="Índice de muestra",
    ylabel="Valor",
    figsize=(12, 6),
    show_scatter=False
):
    y_test = np.asarray(y_test).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_test.shape[0] != y_pred.shape[0]:
        raise ValueError("y_test and y_pred must have the same length.")

    if sort_by_real:
        order = np.argsort(y_test)
        y_test = y_test[order]
        y_pred = y_pred[order]

    if sample_limit is not None:
        y_test = y_test[:sample_limit]
        y_pred = y_pred[:sample_limit]

    x = np.arange(len(y_test))

    plt.figure(figsize=figsize)
    plt.plot(x, y_test, label="Real", linewidth=2)
    plt.plot(x, y_pred, label="Predicción", linewidth=2)

    if show_scatter:
        plt.scatter(x, y_test, s=20)
        plt.scatter(x, y_pred, s=20)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()