from collections import namedtuple
import random

import numpy as np


def boundary_analysis(model, dataset_1, dataset_2, key='embeds_last', k=100, n=100):
    BoundaryAnalysisResult = namedtuple("BoundaryAnalysisResult",[
        "adj_ratio",
        "interp_matrix",
        "bound_results"
    ])
    is_connected = []
    interp_matrix = []
    bound_results = []
    embeds_1 = dataset_1.model_transform(model, key=key)
    embeds_2 = dataset_2.model_transform(model, key=key)
    for _ in range(k):
        sample1 = random.choice(embeds_1)
        sample2 = random.choice(embeds_2)
        interp = []
        min_diff, bound_result = 1, None
        for i in range(1, n):
            result = model(**{key: (sample2 - sample1) * i / n + sample1})
            diff = result['probs'].sort(descending=True)[0][:2].diff().abs().item()
            if diff < min_diff:
                min_diff = diff
                bound_result = result
            interp.append(result["logits"].argmax().item())
        interp_matrix.append(interp)
        bound_results.append(bound_result)
        is_connected.append(np.unique(interp).shape[0] <= 2)
    return BoundaryAnalysisResult(
        adj_ratio=np.mean(is_connected),
        interp_matrix=interp_matrix,
        bound_results=bound_results
    )
