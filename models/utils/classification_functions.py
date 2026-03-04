import matplotlib.pyplot as plt

# Función para ver puntos de un dataset
def plot_points(
    X,
    y=None,
    dims=2,
    elev=25,
    azim=45,
    figsize=(8, 6),
    alpha=0.7,
    title=None,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    legend=True,
    grid=True
):
    if dims not in (2, 3):
        raise ValueError("dims must be either 2 or 3")

    if dims == 2:
        if X.shape[1] < 2:
            raise ValueError("X must have at least 2 columns for 2D plotting")

        plt.figure(figsize=figsize)

        if y is None:
            plt.scatter(X[:, 0], X[:, 1], alpha=alpha)
        else:
            classes = sorted(set(y))
            for c in classes:
                mask = (y == c)
                plt.scatter(X[mask, 0], X[mask, 1], alpha=alpha, label=f"class {c}")

        plt.xlabel(xlabel if xlabel is not None else "x1")
        plt.ylabel(ylabel if ylabel is not None else "x2")
        if title is not None:
            plt.title(title)
        if legend and y is not None:
            plt.legend()
        if grid:
            plt.grid(True)
        plt.show()
        return

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if X.shape[1] < 3:
        raise ValueError("X must have at least 3 columns for 3D plotting")

    if y is None:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=alpha)
    else:
        classes = sorted(set(y))
        for c in classes:
            mask = (y == c)
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], alpha=alpha, label=f"class {c}")

    ax.set_xlabel(xlabel if xlabel is not None else "x1")
    ax.set_ylabel(ylabel if ylabel is not None else "x2")
    ax.set_zlabel(zlabel if zlabel is not None else "x3")
    if title is not None:
        ax.set_title(title)
    if legend and y is not None:
        ax.legend()

    ax.view_init(elev=elev, azim=azim)

    plt.show()



# Función para visualizar test

def plot_true_vs_pred(
    X_test,
    y_true,
    y_pred,
    dims=2,
    elev=25,
    azim=45,
    figsize=(14, 6),
    alpha=0.7,
    title_true="Test (True labels)",
    title_pred="Test (Predicted labels)",
    legend=True,
    grid=True,
    xlabel=None,
    ylabel=None,
    zlabel=None
):
    if dims not in (2, 3):
        raise ValueError("dims must be either 2 or 3")

    X = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test
    yt = y_true.to_numpy() if hasattr(y_true, "to_numpy") else y_true
    yp = y_pred.to_numpy() if hasattr(y_pred, "to_numpy") else y_pred

    if dims == 2:
        if X.shape[1] < 2:
            raise ValueError("X_test must have at least 2 columns for 2D plotting")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for ax, labels, title in [(axes[0], yt, title_true), (axes[1], yp, title_pred)]:
            classes = sorted(set(labels))
            for c in classes:
                mask = (labels == c)
                ax.scatter(X[mask, 0], X[mask, 1], alpha=alpha, label=f"class {c}")
            ax.set_xlabel(xlabel if xlabel is not None else "x1")
            ax.set_ylabel(ylabel if ylabel is not None else "x2")
            ax.set_title(title)
            if grid:
                ax.grid(True)
            if legend:
                ax.legend()

        plt.tight_layout()
        plt.show()
        return

    if X.shape[1] < 3:
        raise ValueError("X_test must have at least 3 columns for 3D plotting")

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, labels, title in [(ax1, yt, title_true), (ax2, yp, title_pred)]:
        classes = sorted(set(labels))
        for c in classes:
            mask = (labels == c)
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], alpha=alpha, label=f"class {c}")

        ax.set_xlabel(xlabel if xlabel is not None else "x1")
        ax.set_ylabel(ylabel if ylabel is not None else "x2")
        ax.set_zlabel(zlabel if zlabel is not None else "x3")
        ax.set_title(title)

        ax.view_init(elev=elev, azim=azim)

        if legend:
            ax.legend()

    plt.tight_layout()
    plt.show()
