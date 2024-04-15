import seaborn as sns
import matplotlib.pyplot as plt
# sns.set_style("white")
sns.set(font_scale=1.6)


def draw_matrix(matrix, names, fmt='.2f', vmin=0, vmax=None,
                annotsize=20, labelsize=18, xlabel='Predicted', ylabel='Actual'):
    ax = sns.heatmap(
        matrix,
        annot=True, annot_kws=dict(size=annotsize), fmt=fmt, vmin=vmin, vmax=vmax, linewidth=1,
        cmap=sns.color_palette("light:b", as_cmap=True),
        xticklabels=names,
        yticklabels=names,
    )
    ax.set_facecolor('white')
    ax.tick_params(axis='x', labelsize=labelsize, rotation=0)
    ax.tick_params(axis='y', labelsize=labelsize, rotation=0)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()
