import numpy as np
import pandas as pd
from plot import plot_real_vs_estimated, plot_residuals
from utility import load_data_csv

def load_coefficients(filename='coefts.csv'):
    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError("Archivo vacío")
        beta = np.array([float(line) for line in lines[:-1]]).reshape(-1, 1)
        lambda_opt = float(lines[-1])
        return beta, lambda_opt
    except Exception as e:
        raise ValueError(f"Error leyendo {filename}: {str(e)}")

def load_selected_vars(filename='selected_vars.csv'):
    try:
        with open(filename, 'r') as f:
            return [int(line.strip()[1:]) - 1 for line in f if line.strip()]
    except Exception as e:
        raise ValueError(f"Error leyendo {filename}: {str(e)}")

def calculate_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    residuals = residuals.flatten()
    dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    return r2, dw

def main():
    print("=== Validación del Modelo Ridge ===")
    try:
        print("\n[1/4] Cargando modelo...")
        beta, lambda_opt = load_coefficients()
        print(f"✓ Coeficientes cargados: {len(beta)} parámetros")
        print(f"✓ Lambda óptimo: {lambda_opt:.6f}")

        print("\n[2/4] Cargando variables...")
        selected_idx = load_selected_vars()
        print(f"✓ Variables seleccionadas: {selected_idx}")

        print("\n[3/4] Cargando datos de test...")
        X_test, y_test = load_data_csv('dtst.csv')
        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test[:, selected_idx]])
        print(f"✓ Datos preparados: {X_test.shape[0]} muestras, {X_test.shape[1]} features")

        print("\n[4/4] Calculando métricas...")
        y_pred = X_test @ beta
        residuals = y_test - y_pred
        r2, dw = calculate_metrics(y_test, y_pred)

        pd.DataFrame({'R2': [r2], 'Durbin-Watson': [dw]}).to_csv('metrica.csv', index=False)
        pd.DataFrame({'Real': y_test.flatten(), 'Estimado': y_pred.flatten()}).to_csv('real_pred.csv', index=False)

        plot_real_vs_estimated(y_test, y_pred, filename='figure3.png')
        plot_residuals(y_pred, residuals, filename='figure4.png')

        print("\n=== Resultados ===")
        print(f"R²: {r2:.4f}")
        print(f"Durbin-Watson: {dw:.4f}")
        print("✓ Métricas guardadas en metrica.csv")
        print("✓ Predicciones guardadas en real_pred.csv")
        print("✓ Gráficos generados: figure3.png y figure4.png")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nSolución:")
        print("1. Ejecute primero trn.py para generar los archivos necesarios")
        print("2. Verifique el formato de los archivos: coefts.csv y selected_vars.csv")
        print("3. Los datos de test deben estar en dtst.csv")

if __name__ == '__main__':
    main()
