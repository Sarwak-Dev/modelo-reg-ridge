import numpy as np
import matplotlib.pyplot as plt

def plot_colinear_index(colinear_index, filename='figure1.png'):
    """Gráfico tipo función Sort() del índice co-lineal ponderado"""
    sorted_indices = np.argsort(colinear_index)
    sorted_values = colinear_index[sorted_indices]

    # Etiquetas con el nombre original de la variable
    variable_labels = [f"X{idx+1}" for idx in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(variable_labels, sorted_values, color='skyblue')
    plt.xlabel('Variables Ordenadas por Índice')
    plt.ylabel('Índice Co-lineal Ponderado')
    plt.title('Función Sort(): Índice Co-lineal Ponderado')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_selected_vars(selected_vars, colinear_index, filename='figure2.png'):
    """Grafica solo las variables seleccionadas (Top-K), ordenadas por su índice"""
    selected_indices = np.array(selected_vars)
    selected_values = colinear_index[selected_indices]
    
    # Ordenar las seleccionadas por índice creciente
    order = np.argsort(selected_values)
    selected_indices = selected_indices[order]
    selected_values = selected_values[order]

    plt.figure(figsize=(10, 6))
    plt.bar([f"X{i+1}" for i in selected_indices], selected_values, color='green')
    plt.xlabel('Variables Seleccionadas (Top-K)')
    plt.ylabel('Índice Co-lineal Ponderado')
    plt.title('Top-K Variables Seleccionadas (Ordenadas)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_real_vs_estimated(y_real, y_estimate, filename='figure3.png'):
    """Valores reales y estimados sobre las muestras"""
    plt.figure(figsize=(10, 6))
    plt.plot(y_real, label='Valores Reales', marker='o')
    plt.plot(y_estimate, label='Valores Estimados', marker='x')
    plt.xlabel('Muestra')
    plt.ylabel('Valor')
    plt.title('Valores Reales vs Estimados')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_residuals(y_estimate, residuals, filename='figure4.png'):
    """Residuos vs valores estimados"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_estimate, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Valores Estimados')
    plt.ylabel('Residuos')
    plt.title('Residuos vs Valores Estimados')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_gcv(lambdas, gcv_values, optimal_lambda, filename='gcv_curve.png'):
    """Grafica la curva GCV con el lambda óptimo marcado"""
    plt.figure(figsize=(10, 6))
    plt.semilogx(lambdas, gcv_values, label='GCV')
    plt.axvline(x=optimal_lambda, color='r', linestyle='--', label=f'λ óptimo = {optimal_lambda:.4f}')
    plt.xlabel('Lambda (escala logarítmica)')
    plt.ylabel('Valor GCV')
    plt.title('Curva GCV para Selección de Lambda Óptimo')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
