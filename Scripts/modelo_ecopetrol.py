
# ============================================================
# ECONOMETRIA II: ANALISIS DE LOS LOG-RETORNOS DE ECOPETROL
# Carga de datos, estadistica descriptiva, graficas y prueba ADF
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


# ------------------------------------------------------------
# 1. RUTAS DEL PROYECTO
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "ecopetrol_weekly_clean.csv"
OUTPUT_DIR = ROOT / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. CARGA Y PREPARACION DE LOS DATOS
# ------------------------------------------------------------

print("=" * 65)
print("TRABAJO DE ECONOMETRIA II: ACCIONES DE ECOPETROL")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"No se encontro la base de datos en: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

# Verificar las columnas necesarias
required_columns = {"date", "close", "ret"}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Faltan columnas en el CSV: {sorted(missing_columns)}"
    )

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["ret"] = pd.to_numeric(df["ret"], errors="coerce")

df = (
    df.dropna(subset=["date"])
      .sort_values("date")
      .reset_index(drop=True)
)

returns = df.dropna(subset=["ret"]).copy()
serie = returns["ret"].astype(float)

if len(serie) < 10:
    raise ValueError("Hay muy pocos retornos validos para el analisis.")


# ------------------------------------------------------------
# 3. INFORMACION GENERAL Y ESTADISTICAS DESCRIPTIVAS
# ------------------------------------------------------------

print(f"\nArchivo: {DATA_PATH.name}")
print(f"Numero de filas: {len(df)}")
print(f"Fecha inicial: {df['date'].min().date()}")
print(f"Fecha final: {df['date'].max().date()}")
print(f"Retornos validos: {len(serie)}")
print(f"Retornos faltantes: {df['ret'].isna().sum()}")

estadisticas = serie.describe()

print("\nESTADISTICAS DESCRIPTIVAS DE LOS LOG-RETORNOS")
print(estadisticas.to_string())

estadisticas.to_csv(
    OUTPUT_DIR / "estadisticas_descriptivas.csv",
    header=["valor"]
)

print("\nEstadisticas guardadas en Outputs.")


# ------------------------------------------------------------
# 4. GRAFICA DE LA SERIE DE LOG-RETORNOS
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    returns["date"],
    returns["ret"],
    linewidth=0.8
)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_title("Ecopetrol: log-retornos semanales")
ax.set_xlabel("Fecha")
ax.set_ylabel("Log-retorno")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "log_retornos_ecopetrol.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("\nGrafica temporal guardada en Outputs.")


# ------------------------------------------------------------
# 5. HISTOGRAMA DE LOS LOG-RETORNOS
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    serie,
    bins=40,
    edgecolor="black"
)
ax.axvline(
    serie.mean(),
    color="red",
    linestyle="--",
    label="Media"
)
ax.set_title("Distribucion de los log-retornos semanales")
ax.set_xlabel("Log-retorno")
ax.set_ylabel("Frecuencia")
ax.legend()

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "distribucion_retornos.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Histograma guardado en Outputs.")


# ------------------------------------------------------------
# 6. PRUEBA DICKEY-FULLER AUMENTADA (ADF)
# ------------------------------------------------------------

# Especificacion con constante y seleccion automatica de rezagos
resultado_adf = adfuller(serie, regression="c", autolag="AIC")

estadistico_adf = resultado_adf[0]
valor_p = resultado_adf[1]
rezagos = resultado_adf[2]
observaciones = resultado_adf[3]
valores_criticos = resultado_adf[4]

print("\n" + "=" * 65)
print("PRUEBA DE ESTACIONARIEDAD: DICKEY-FULLER AUMENTADA")
print("=" * 65)

print(f"Estadistico ADF: {estadistico_adf:.6f}")
print(f"Valor p:         {valor_p:.6f}")
print(f"Rezagos usados:  {rezagos}")
print(f"Observaciones:   {observaciones}")

print("\nValores criticos:")
for nivel, valor in valores_criticos.items():
    print(f"{nivel}: {valor:.6f}")

if valor_p < 0.05:
    print("\nConclusion al 5%:")
    print("Se rechaza la hipotesis nula de raiz unitaria.")
    print("La prueba aporta evidencia de estacionariedad.")
else:
    print("\nConclusion al 5%:")
    print("No se rechaza la hipotesis nula de raiz unitaria.")
    print("La prueba no aporta evidencia suficiente de estacionariedad.")


# ------------------------------------------------------------
# 7. GUARDAR RESULTADOS DE LA PRUEBA ADF
# ------------------------------------------------------------

resultados_adf = pd.DataFrame({
    "indicador": [
        "Estadistico ADF",
        "Valor p",
        "Rezagos usados",
        "Observaciones"
    ],
    "valor": [
        estadistico_adf,
        valor_p,
        rezagos,
        observaciones
    ]
})

resultados_adf.to_csv(
    OUTPUT_DIR / "resultado_prueba_adf.csv",
    index=False
)

print("\nResultados ADF guardados en Outputs.")
print("\nANALISIS INICIAL COMPLETADO.")
print(f"Carpeta de resultados: {OUTPUT_DIR}")







# ============================================================
# 8. IDENTIFICACION INICIAL: ACF Y PACF
# ============================================================

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Numero de rezagos para las graficas
nlags = 30

# ACF
fig, ax = plt.subplots(figsize=(11, 5))
plot_acf(serie, lags=nlags, zero=True, ax=ax)
ax.set_title("ACF de los log-retornos semanales de Ecopetrol")
ax.set_xlabel("Rezago")
ax.set_ylabel("Autocorrelacion")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "acf_ecopetrol.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# PACF
fig, ax = plt.subplots(figsize=(11, 5))
plot_pacf(serie, lags=nlags, method="ywm", ax=ax)
ax.set_title("PACF de los log-retornos semanales de Ecopetrol")
ax.set_xlabel("Rezago")
ax.set_ylabel("Autocorrelacion parcial")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "pacf_ecopetrol.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("\nACF y PACF guardadas en Outputs.")
print("Archivos: acf_ecopetrol.png y pacf_ecopetrol.png")
