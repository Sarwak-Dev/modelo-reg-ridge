import numpy as np

def ridge_gcv(X, y, lambdas):
    """Calcula GCV para regresión Ridge usando SVD"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    n = len(y)
    gcv_scores = []
    
    for lam in lambdas:
        d = s**2 / (s**2 + lam)
        y_hat = U @ (d * (U.T @ y))
        rss = np.sum((y - y_hat)**2)
        df = np.sum(d)
        gcv = (rss/n) / (1 - df/n)**2
        gcv_scores.append(gcv)
    
    idx = np.argmin(gcv_scores)
    return lambdas[idx], gcv_scores

def ridge_beta(X, y, lam):
    """Calcula coeficientes Ridge usando SVD (más estable)"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    d = s / (s**2 + lam)
    return Vt.T @ (d * (U.T @ y))

def compute_vif(X):
    """Calcula VIF para cada variable"""
    p = X.shape[1]
    vifs = np.zeros(p)
    
    for i in range(p):
        X_temp = np.delete(X, i, axis=1)
        y_temp = X[:, i]
        beta = np.linalg.pinv(X_temp) @ y_temp
        y_pred = X_temp @ beta
        rss = np.sum((y_temp - y_pred)**2)
        tss = np.sum((y_temp - np.mean(y_temp))**2)
        r2 = 1 - rss/tss if tss != 0 else 1
        vifs[i] = 1/(1 - r2) if r2 != 1 else np.inf
    
    return vifs

def colineal_index(X, beta, lambda_opt):
    """Calcula índice co-lineal ponderado según especificaciones del PDF"""
    vifs = compute_vif(X)
    P = np.mean(vifs) / vifs
    
    var_beta = np.var(beta)
    mean_var = np.mean(var_beta) if isinstance(var_beta, np.ndarray) else var_beta
    Q = var_beta / mean_var if mean_var != 0 else np.ones_like(var_beta)
    
    return P * Q