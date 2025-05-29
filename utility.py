import numpy as np
import pandas as pd

def load_data_csv(nfile):
    data = pd.read_csv(nfile)
    X = data.iloc[:, :-1].values.astype(float)
    y = data.iloc[:, -1].values.astype(float).reshape(-1, 1)
    return X, y

def split_data(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def svd_decomposition(X):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U, s, Vt

def pinv_svd(X, lambda_val=0):
    U, s, Vt = svd_decomposition(X)
    s_lambda = s / (s**2 + lambda_val)
    return Vt.T @ np.diag(s_lambda) @ U.T

def calculate_gcv(X, y, lambda_val):
    n = X.shape[0]
    U, s, _ = svd_decomposition(X)
    y_hat = np.zeros_like(y)
    f_lambda = 0
    for i in range(len(s)):
        u_i = U[:, i].reshape(-1, 1)
        s_i = s[i]
        factor = s_i**2 / (s_i**2 + lambda_val)
        y_hat += factor * (u_i.T @ y) * u_i
        f_lambda += factor
    mse = np.mean((y - y_hat)**2)
    gcv = mse / (1 - f_lambda / n)**2
    return gcv

def find_optimal_lambda(X, y, lambda_min=0.0001, lambda_max=10000, n_lambdas=100):
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambdas)
    gcv_values = [calculate_gcv(X, y, l) for l in lambdas]
    optimal_lambda = lambdas[np.argmin(gcv_values)]
    return optimal_lambda, lambdas, gcv_values

def ridge_regression(X, y, lambda_val):
    n_features = X.shape[1]
    XtX = X.T @ X
    XtX_ridge = XtX + lambda_val * np.eye(n_features)
    coefficients = np.linalg.solve(XtX_ridge, X.T @ y)
    return coefficients

def calculate_vif(X):
    vif = []
    n_features = X.shape[1]
    for i in range(n_features):
        X_temp = np.delete(X, i, axis=1)
        y_temp = X[:, i].reshape(-1, 1)
        X_temp = np.hstack([np.ones((X_temp.shape[0], 1)), X_temp])
        beta = np.linalg.inv(X_temp.T @ X_temp) @ X_temp.T @ y_temp
        y_pred = X_temp @ beta
        ss_res = np.sum((y_temp - y_pred)**2)
        ss_tot = np.sum((y_temp - np.mean(y_temp))**2)
        r_squared = 1 - (ss_res / ss_tot)
        vif.append(1. / (1. - r_squared) if r_squared < 1 else float('inf'))
    return np.array(vif)

def weighted_colinear_index(X, y, lambda_val):
    vif = calculate_vif(X)
    P = np.mean(vif)
    beta = ridge_regression(X, y, lambda_val)
    X_ridge = X.T @ X + lambda_val * np.eye(X.shape[1])
    cov_beta = np.linalg.inv(X_ridge)
    var_beta = np.diag(cov_beta)
    Q = np.mean(var_beta)
    I = P * Q * vif * var_beta
    return I, vif, var_beta

def save_coefficients(beta, lambda_opt, filename='coefts.csv'):
    with open(filename, 'w') as f:
        for coef in beta.flatten():
            f.write(f"{coef:.8f}\n")
        f.write(f"{lambda_opt:.8f}\n")

def save_selected_vars(selected_vars, filename='selected_vars.csv'):
    pd.DataFrame(selected_vars).to_csv(filename, index=False, header=False)
