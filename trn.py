import numpy as np
import pandas as pd
from utility import *
from plot import plot_colinear_index, plot_selected_vars, plot_gcv

def selection_vars(X, y, lambda_opt, top_k=8):
    """Selecciona las top_k variables usando el índice co-lineal ponderado"""
    n_vars = X.shape[1]
    remaining_vars = list(range(n_vars))
    selected_vars = []
    colinear_indices = []

    I, vif, var_beta = weighted_colinear_index(X, y, lambda_opt)
    colinear_indices.append(I.copy())

    for _ in range(top_k):
        min_idx = np.argmin(I[remaining_vars])
        selected_var = remaining_vars[min_idx]
        selected_vars.append(selected_var)
        remaining_vars.remove(selected_var)

        if remaining_vars:
            X_remaining = X[:, remaining_vars]
            I_new, _, _ = weighted_colinear_index(X_remaining, y, lambda_opt)
            I[remaining_vars] = I_new

    return selected_vars, colinear_indices[0]

def main():
    # 1. Cargar datos
    X, y = load_data_csv('dataset.csv')

    # 2. Dividir datos (80% train, 20% test)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 3. Guardar conjuntos de train y test
    pd.DataFrame(np.hstack([X_train, y_train])).to_csv('dtrn.csv', index=False, header=False)
    pd.DataFrame(np.hstack([X_test, y_test])).to_csv('dtst.csv', index=False, header=False)

    # 4. Encontrar lambda óptimo
    lambda_opt, lambdas, gcv_values = find_optimal_lambda(X_train, y_train)

    # 5. Seleccionar variables
    selected_vars, colinear_index = selection_vars(X_train, y_train, lambda_opt)

    # 6. Preparar datos con intercepto
    X_train_selected = np.hstack([np.ones((X_train.shape[0], 1)), X_train[:, selected_vars]])

    # 7. Entrenar modelo final
    beta = ridge_regression(X_train_selected, y_train, lambda_opt)

    # 8. Guardar resultados
    save_coefficients(beta, lambda_opt)
    save_selected_vars([f'X{i+1}' for i in selected_vars])

    # 9. Generar gráficos
    plot_colinear_index(colinear_index)
    plot_selected_vars(selected_vars, colinear_index)
    plot_gcv(lambdas, gcv_values, lambda_opt)

if __name__ == '__main__':
    main()
