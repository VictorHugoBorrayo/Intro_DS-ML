import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_real_vs_pred(
    y_test,
    y_pred,
    sample_limit=None,
    sort_by_real=False,
    title="Valores reales vs predicción",
    xlabel="x",
    ylabel="Valor",
    figsize=(12, 6),
    show_scatter=False,
    x_values=None,
    x_col=None
):
    y_test = np.asarray(y_test).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_test.shape[0] != y_pred.shape[0]:
        raise ValueError("y_test and y_pred must have the same length.")

    if x_values is None:
        raise ValueError("x_values must be provided.")

    if isinstance(x_values, pd.DataFrame):
        if x_values.shape[1] == 1:
            x_plot = x_values.iloc[:, 0].to_numpy().reshape(-1)
            if x_col is None:
                x_col = x_values.columns[0]
        else:
            if x_col is None:
                raise ValueError("For multi-feature DataFrames, x_col must be specified.")
            x_plot = x_values[x_col].to_numpy().reshape(-1)

    elif isinstance(x_values, pd.Series):
        x_plot = x_values.to_numpy().reshape(-1)
        if x_col is None:
            x_col = x_values.name

    else:
        x_plot = np.asarray(x_values)
        if x_plot.ndim == 2:
            if x_plot.shape[1] == 1:
                x_plot = x_plot[:, 0]
            else:
                raise ValueError("For multi-feature arrays, provide a single x column.")
        x_plot = x_plot.reshape(-1)

    if x_plot.shape[0] != y_test.shape[0]:
        raise ValueError("x_values must have the same number of rows as y_test.")

    if sort_by_real:
        order = np.argsort(y_test)
    else:
        order = np.argsort(x_plot)

    x_plot = x_plot[order]
    y_test = y_test[order]
    y_pred = y_pred[order]

    if sample_limit is not None:
        x_plot = x_plot[:sample_limit]
        y_test = y_test[:sample_limit]
        y_pred = y_pred[:sample_limit]

    plt.figure(figsize=figsize)

    if show_scatter:
        plt.scatter(x_plot, y_test, s=20, label="Real")
        plt.scatter(x_plot, y_pred, s=20, label="Predicción")
    else:
        plt.plot(x_plot, y_test, label="Real", linewidth=2)
        plt.plot(x_plot, y_pred, label="Predicción", linewidth=2)

    plt.title(title)
    plt.xlabel(x_col if x_col is not None else xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()