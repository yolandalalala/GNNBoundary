from collections import defaultdict
from itertools import combinations_with_replacement

import numpy as np

from .boundary_analysis import boundary_analysis


def pairwise_boundary_analysis(model, dataset_list, key='embeds_last', k=100, n=100):
    c = len(dataset_list)
    adj_ratio_mat = np.zeros((c, c))
    result_dict = defaultdict(dict)
    for i, j in combinations_with_replacement(range(c), 2):
        result_dict[i][j] = result_dict[j][i] = result = boundary_analysis(
            model, dataset_list[i], dataset_list[j],
            key=key, k=k, n=n
        )
        adj_ratio_mat[i, j] = adj_ratio_mat[j, i] = result.adj_ratio
    return adj_ratio_mat, result_dict