import pandas as pd
import numpy as np
from pathlib import Path

# Cargar el archivo CSV

BASE_DIR = Path(__file__).resolve().parent.parent
archivo_csv = BASE_DIR / "outputs" / "hierarchical_clusters_full.csv"
df_base = pd.read_csv(archivo_csv, encoding='latin1', sep=';')



# Verificar columnas disponibles (opcional)
print("Columnas disponibles:", df_base.columns.tolist())

# === 2. Variables 
variables = [
    'swd', 'ws', 'pedFlow', 'pvQ', 'pedInfr', 'Obstr',
    'Pres_pol_c', 'Crime', 'Crashes', 'Lighting', 'Cs',
    'ptStops', 'Tcontrl', 'trFlow', 'crTime', 'trDens',
    'insDens', 'ResDens', 'Aesthetics', 'Trees', 'Block',
    'noise', 'StagH2o', 'Slope', 'AirPol', 'Cleanless',
    'mpgaDens'
]

# Asegurarnos de que todas las variables estén en el DataFrame
faltantes = [v for v in variables if v not in df_base.columns]
if faltantes:
    raise KeyError(f"Las siguientes columnas faltan en el CSV: {faltantes}")


df = (
    df_base
    .groupby('Cluster_Jerarquico')[variables]
    .mean()
    .reset_index()
)



# === 3. Definir qué variables son 'coste' (a invertir) ===
cost_vars = [
    'Crime', 'Crashes',
    'StagH2o', 'Cs',
    'trFlow', 'Block'
]
# Las demás se asumen 'beneficio'
benefit_vars = [v for v in variables if v not in cost_vars]

# === 4. Extraer los valores por cluster y normalizar para obtener pesos ===
# Asegurarnos de que 'Cluster_Jerarquico' existe
if 'Cluster_Jerarquico' not in df.columns:
    raise KeyError("No se encontró la columna 'Cluster_Jerarquico' en el CSV.")

# Seleccionar solo las columnas de interés y usar 'Cluster_Jerarquico' como índice
V = df.set_index('Cluster_Jerarquico')[variables].copy()

# Normalizamos fila a fila para que cada cluster tenga pesos que sumen 1
W = V.div(V.sum(axis=1), axis=0)

# === 5. Generar las fórmulas de CI considerando inversión de los costes ===
for cluster_id, row in W.iterrows():
    terms = []
    for var in variables:
        peso_pct = row[var] * 100
        weight_str = f"{peso_pct:.2f}"
        if var in cost_vars:
            # Invertimos la escala: (1 - x_var)
            terms.append(f"{weight_str}(1 - {var})")
        else:
            terms.append(f"{weight_str}{var}")
    formula = " + ".join(terms)
    print(f"CI_{cluster_id}(x) = {formula}")
