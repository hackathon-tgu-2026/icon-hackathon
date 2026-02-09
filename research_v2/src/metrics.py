import numpy as np
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr

def normalize_ranking(ranking):
    n = len(ranking)
    return [n - r + 1 for r in ranking]

def compute_metrics(hr_rank, model_rank):
    """
    hr_rank: список номеров резюме в порядке предпочтения
    model_rank: список номеров резюме в порядке предпочтения
    """

    # NDCG
    hr_scores = normalize_ranking(hr_rank)
    model_scores = normalize_ranking(model_rank)
    ndcg = ndcg_score([hr_scores], [model_scores])

    # Spearman
    # Если один из списков константный или длина < 2, ставим 0
    if len(set(hr_rank)) < 2 or len(set(model_rank)) < 2:
        spearman = 0.0
    else:
        spearman = spearmanr(hr_rank, model_rank).correlation
        # На всякий случай, если scipy вернул nan
        if spearman is None or np.isnan(spearman):
            spearman = 0.0

    return ndcg, spearman