import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer


def business_cost(y_true, y_proba, threshold=0.5, cost_fn=10, cost_fp=1):
    """Calcule le coût métier total.

    FN = mauvais client prédit bon client : coût élevé.
    FP = bon client prédit mauvais client : coût plus faible.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return cost_fn * fn + cost_fp * fp


def find_best_threshold(y_true, y_proba, cost_fn=10, cost_fp=1):
    """Teste plusieurs seuils et retourne celui qui minimise le coût métier."""
    thresholds = np.arange(0.01, 0.99, 0.01)
    costs = [business_cost(y_true, y_proba, t, cost_fn, cost_fp) for t in thresholds]
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])
