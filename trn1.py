import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    """Carga los archivos de datos y configuración"""
    data = pd.read_csv("dataset.csv").values
    cfg = pd.read_csv("cfg_lambda.csv")
    return data, cfg.iloc[0, 0], cfg.iloc[0, 1], int(cfg.iloc[0, 2])

def prepare_datasets(data):
    """Divide y normaliza los datos"""
    X, y = data[:, :-1], data[:, -1].reshape(-1, 1)
    idx = np.random.permutation(len(X))
    n_trn = int(0.8 * len(X))
    
    Xtr, ytr = X[idx[:n_trn]], y[idx[:n_trn]]
    Xtst, ytst = X[idx[n_trn:]], y[idx[n_trn:]]
    
    # Normalización
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0)
    Xtr = (Xtr - mean) / std
    Xtst = (Xtst - mean) / std
    
    # Guardar datasets
    np.savetxt("dtrn.csv", np.hstack([Xtr, ytr]), delimiter=",")
    np.savetxt("dtst.csv", np.hstack([Xtst, ytst]), delimiter=",")
    
    return Xtr, ytr, Xtst, ytst

def ridge_regression(X, y, lambda_):
    """Calcula los coeficientes de regresión Ridge"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    d = s / (s**2 + lambda_)
    return Vt.T @ (d * (U.T @ y))

def compute_vif(X):
    """Calcula el Factor de Inflación de Varianza para cada variable"""
    vifs = []
    for i in range(X.shape[1]):
        X_temp = np.delete(X, i, axis=1)
        y_temp = X[:, i]
        beta = np.linalg.pinv(X_temp) @ y_temp
        y_pred = X_temp @ beta
        rss = np.sum((y_temp - y_pred)**2)
        tss = np.sum((y_temp - np.mean(y_temp))**2)
        r2 = 1 - rss/tss if tss != 0 else 1
        vifs.append(1/(1 - r2) if r2 != 1 else np.inf)
    return np.array(vifs)

def colineal_index(X, beta, lambda_):
    """Calcula el índice co-lineal ponderado"""
    vifs = compute_vif(X)
    P = np.mean(vifs) / vifs
    
    var_beta = np.var(beta)
    Q = var_beta / np.mean(var_beta) if np.mean(var_beta) != 0 else 1.0
    
    return P * Q

def select_variables(X, y, lambda_opt, TopK):
    """Selecciona las variables más relevantes"""
    variables = list(range(X.shape[1]))
    while len(variables) > TopK:
        Xtmp = X[:, variables]
        beta = ridge_regression(Xtmp, y, lambda_opt)
        idx_vals = colineal_index(Xtmp, beta, lambda_opt)
        worst = np.argmax(idx_vals)
        del variables[worst]
    return variables

def plot_figure1(var_names, index_values, selected_vars):
    """Genera el gráfico de barras del índice co-lineal (Figura 1)"""
    plt.figure(figsize=(12, 6))
    
    # Ordenar por valor del índice
    sorted_idx = np.argsort(index_values)
    sorted_names = np.array(var_names)[sorted_idx]
    sorted_values = index_values[sorted_idx]
    
    # Crear barras con colores diferenciados
    colors = ['limegreen' if name in selected_vars else 'skyblue' for name in sorted_names]
    plt.bar(sorted_names, sorted_values, color=colors)
    
    # Configuración del gráfico
    plt.xlabel('Variables', fontsize=12)
    plt.ylabel('Índice Co-lineal Ponderado', fontsize=12)
    plt.title('Índice Co-lineal Ponderado por Variable', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Línea de promedio
    mean_val = np.mean(index_values)
    plt.axhline(mean_val, color='red', linestyle='--', label=f'Promedio: {mean_val:.2f}')
    plt.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figure1.png', dpi=300)
    plt.close()

def plot_figure2(var_names, index_values, selected_vars):
    """Genera el gráfico de variables seleccionadas (Figura 2)"""
    selected_idx = [i for i, name in enumerate(var_names) if name in selected_vars]
    selected_values = index_values[selected_idx]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(np.array(var_names)[selected_idx], selected_values, color='limegreen')
    
    # Añadir valores encima de las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.xlabel('Variables Seleccionadas', fontsize=12)
    plt.ylabel('Índice Co-lineal', fontsize=12)
    plt.title(f'Top {len(selected_vars)} Variables Seleccionadas', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figure2.png', dpi=300)
    plt.close()

def main():
    # 1. Cargar y preparar datos
    data, lamb_min, lamb_max, TopK = load_data()
    Xtr, ytr, Xtst, ytst = prepare_datasets(data)
    
    # 2. Calcular lambda óptimo (simplificado)
    lambda_opt = (lamb_min + lamb_max) / 2  # Valor medio del rango
    
    # 3. Calcular índice co-lineal para todas las variables
    beta_all = ridge_regression(Xtr, ytr, lambda_opt)
    index_all = colineal_index(Xtr, beta_all, lambda_opt)
    var_names = [f"X{i+1}" for i in range(Xtr.shape[1])]
    
    # 4. Selección de variables
    selected_vars_idx = select_variables(Xtr, ytr, lambda_opt, TopK)
    selected_names = [f"X{i+1}" for i in selected_vars_idx]
    pd.Series(selected_names).to_csv("selected_vars.csv", index=False, header=False)
    
    # 5. Calcular coeficientes finales
    Xsel = Xtr[:, selected_vars_idx]
    beta_final = ridge_regression(Xsel, ytr, lambda_opt)
    np.savetxt("coefts.csv", np.append(beta_final, lambda_opt)[None], delimiter=",")
    
    # 6. Generar gráficos requeridos
    plot_figure1(var_names, index_all, selected_names)
    plot_figure2(var_names, index_all, selected_names)

if __name__ == "__main__":
    main()