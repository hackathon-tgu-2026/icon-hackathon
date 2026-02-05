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
    n = len(hr_rank)

    hr_scores = normalize_ranking(hr_rank)
    model_scores = normalize_ranking(model_rank)

    ndcg = ndcg_score([hr_scores], [model_scores])
    spearman = spearmanr(hr_rank, model_rank).correlation

    return ndcg, spearman