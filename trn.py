import numpy as np
import pandas as pd
from utility import *
from plot import plot_colinear_index, plot_selected_vars, plot_gcv

def selection_vars(X, y, lambda_opt, top_k=8):
    n_vars = X.shape[1]
    remaining_vars = list(range(n_vars))
    I_full = None  # para graficar

    for _ in range(n_vars - top_k):
        I, _, _ = weighted_colinear_index(X[:, remaining_vars], y, lambda_opt)
        idx_max = np.argmax(I)
        del remaining_vars[idx_max]
        I_full = I  # el último I calculado

    # Creamos vector I_global con tamaño total
    I_plot = np.full(n_vars, np.nan)
    for i, idx in enumerate(remaining_vars):
        I_plot[idx] = I_full[i]  # colocamos valores en posición original

    return remaining_vars, I_plot  # ← ahora puedes graficar sin error

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
