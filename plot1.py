import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar coeficientes y variables
coefs = pd.read_csv("coefts.csv", header=None).values.flatten()
beta = coefs[:-1]
lambda_opt = coefs[-1]

var_names = pd.read_csv("selected_vars.csv", header=None).values.flatten()

# Graficar
plt.figure()
plt.bar(var_names, beta)
plt.title(f"Figura 2: Coeficientes Ridge (λ = {lambda_opt:.4f})")
plt.ylabel("Coeficiente")
plt.xlabel("Variable")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figure2.png")
