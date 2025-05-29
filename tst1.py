import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos
test_data = pd.read_csv("dtst.csv", header=None).values
selected = pd.read_csv("selected_vars.csv", header=None).values.flatten()
coefs = pd.read_csv("coefts.csv", header=None).values.flatten()
beta = coefs[:-1]

Xtest = test_data[:, :-1]
yreal = test_data[:, -1]
idx_vars = [int(v[1:]) - 1 for v in selected]
Xsel = Xtest[:, idx_vars]

# Predicción
ypred = Xsel @ beta
residuals = yreal - ypred

# Guardar CSVs
pd.DataFrame({'real': yreal, 'pred': ypred}).to_csv("real_pred.csv", index=False)
r2 = 1 - np.sum(residuals**2) / np.sum((yreal - np.mean(yreal))**2)
dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
pd.DataFrame({'R2': [r2], 'DurbinWatson': [dw]}).to_csv("metrica.csv", index=False)

# Graficar Real vs Predicho
plt.figure()
plt.plot(yreal, label="Real")
plt.plot(ypred, label="Estimado")
plt.legend()
plt.title("Figura 3: Valores Reales vs Estimados")
plt.savefig("figure3.png")

# Graficar Residuos
plt.figure()
plt.scatter(ypred, residuals)
plt.axhline(0, color='r', linestyle='--')
plt.title("Figura 4: Residuos vs Estimado")
plt.xlabel("Valor Estimado")
plt.ylabel("Residual")
plt.savefig("figure4.png")
