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










# ============================================================
# DIAGNOSTICO: RUIDO BLANCO Y DEPENDENCIA EN LA VOLATILIDAD
# ============================================================

from statsmodels.stats.diagnostic import acorr_ljungbox

print("\n" + "=" * 60)
print("PRUEBAS DE LJUNG-BOX")
print("=" * 60)

# Prueba sobre los log-retornos
lb_retornos = acorr_ljungbox(
    serie,
    lags=[5, 10, 20],
    return_df=True
)

print("\nLjung-Box sobre log-retornos:")
print(lb_retornos)

# Prueba sobre los retornos al cuadrado
lb_cuadrados = acorr_ljungbox(
    serie ** 2,
    lags=[5, 10, 20],
    return_df=True
)

print("\nLjung-Box sobre log-retornos al cuadrado:")
print(lb_cuadrados)

# Interpretacion automatica
for rezago, fila in lb_retornos.iterrows():
    if fila["lb_pvalue"] < 0.05:
        print(
            f"Retornos, lag {rezago}: evidencia de "
            "autocorrelacion conjunta al 5%."
        )
    else:
        print(
            f"Retornos, lag {rezago}: no se rechaza "
            "la ausencia de autocorrelacion al 5%."
        )

for rezago, fila in lb_cuadrados.iterrows():
    if fila["lb_pvalue"] < 0.05:
        print(
            f"Cuadrados, lag {rezago}: evidencia de "
            "dependencia en la magnitud de los retornos."
        )
    else:
        print(
            f"Cuadrados, lag {rezago}: no se detecta "
            "dependencia conjunta significativa al 5%."
        )





# ============================================================
# PRUEBA ARCH-LM: DEPENDENCIA EN LA VARIANZA
# ============================================================

from statsmodels.stats.diagnostic import het_arch

print("\n" + "=" * 60)
print("PRUEBA ARCH-LM SOBRE LOS LOG-RETORNOS")
print("=" * 60)

# La prueba se aplica inicialmente a los retornos centrados
retornos_centrados = serie - serie.mean()

arch_lm, arch_pvalue, f_stat, f_pvalue = het_arch(
    retornos_centrados,
    nlags=10
)

print(f"Estadistico LM: {arch_lm:.6f}")
print(f"Valor p LM:     {arch_pvalue:.8g}")
print(f"Estadistico F:  {f_stat:.6f}")
print(f"Valor p F:      {f_pvalue:.8g}")

if arch_pvalue < 0.05:
    print("\nSe rechaza H0 al 5%.")
    print("Hay evidencia de efectos ARCH.")
    print("Se justifica evaluar modelos ARCH/GARCH.")
else:
    print("\nNo se rechaza H0 al 5%.")
    print("No hay evidencia suficiente de efectos ARCH")
    print("con esta especificacion y numero de rezagos.")









# ============================================================
# 10. ESTIMACION Y COMPARACION DE MODELOS ARMA
# ============================================================

import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

print("\n" + "=" * 65)
print("COMPARACION DE MODELOS ARMA PARA ECOPETROL")
print("=" * 65)

# Utilizar la serie de log-retornos ya preparada.
# No se recortan fechas ni se eliminan valores extremos.
y = serie.dropna().astype(float)

# Candidatos iniciales ARMA(p,q)
candidatos = [
    (1, 0),
    (2, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 2),
]

resultados = {}
filas = []

for p, q in candidatos:

    nombre = f"ARMA({p},{q})"

    try:
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")

            modelo = ARIMA(
                y,
                order=(p, 0, q),
                trend="c"
            )

            ajuste = modelo.fit()

        # Comprobar convergencia cuando el optimizador
        # informa este dato.
        convergio = ajuste.mle_retvals.get(
            "converged", None
        )

        # Ljung-Box sobre residuos, ajustando los grados
        # de libertad por los parametros AR y MA.
        residuos = ajuste.resid

        diagnostico = acorr_ljungbox(
            residuos,
            lags=[10, 20],
            model_df=p + q,
            return_df=True
        )

        p_lb10 = diagnostico.loc[10, "lb_pvalue"]
        p_lb20 = diagnostico.loc[20, "lb_pvalue"]

        # ARCH-LM sobre residuos del modelo de media
        from statsmodels.stats.diagnostic import het_arch

        arch_lm, arch_p, arch_f, arch_fp = het_arch(
            residuos,
            nlags=10,
            ddof=p + q
        )

        resultados[nombre] = ajuste

        filas.append({
            "modelo": nombre,
            "p": p,
            "q": q,
            "AIC": ajuste.aic,
            "BIC": ajuste.bic,
            "LogLik": ajuste.llf,
            "convergencia": convergio,
            "p_LjungBox_10": p_lb10,
            "p_LjungBox_20": p_lb20,
            "p_ARCH_LM": arch_p,
            "parametros_significativos": int(
                (ajuste.pvalues < 0.05).sum()
            ),
        })

        print(
            f"{nombre}: AIC={ajuste.aic:.3f}; "
            f"BIC={ajuste.bic:.3f}; "
            f"Ljung-Box(10) p={p_lb10:.4g}; "
            f"ARCH-LM p={arch_p:.4g}"
        )

    except Exception as error:
        print(f"Error al estimar {nombre}: {error}")


# Guardar la tabla comparativa
tabla_arma = pd.DataFrame(filas)

if tabla_arma.empty:
    raise RuntimeError(
        "No se pudo estimar ningun candidato ARMA."
    )

tabla_arma = tabla_arma.sort_values(
    "BIC"
).reset_index(drop=True)

tabla_arma.to_csv(
    OUTPUT_DIR / "comparacion_modelos_arma.csv",
    index=False
)

print("\n" + "=" * 65)
print("RESULTADOS ORDENADOS POR BIC")
print("=" * 65)

print(tabla_arma.to_string(index=False))

print(
    "\nTabla guardada en "
    "Outputs/comparacion_modelos_arma.csv"
)

# Resumen del candidato con menor BIC
mejor_nombre = tabla_arma.iloc[0]["modelo"]
mejor_ajuste = resultados[mejor_nombre]

print(f"\nCandidato con menor BIC: {mejor_nombre}")
print(mejor_ajuste.summary())

# Guardar residuos y graficas del candidato inicial
residuos_mejor = mejor_ajuste.resid

pd.DataFrame({
    "residuo": residuos_mejor
}).to_csv(
    OUTPUT_DIR / "residuos_mejor_arma.csv",
    index=False
)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(
    residuos_mejor,
    lags=30,
    ax=axes[0],
    zero=False
)
axes[0].set_title(
    f"ACF de residuos: {mejor_nombre}"
)

plot_pacf(
    residuos_mejor,
    lags=30,
    ax=axes[1],
    method="ywm",
    zero=False
)
axes[1].set_title(
    f"PACF de residuos: {mejor_nombre}"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "diagnostico_residuos_arma.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nDiagnostico de residuos guardado en Outputs.")
print("\nCOMPARACION ARMA COMPLETADA.")






# ============================================================
# SELECCION AUTOMATICA DEL MODELO ARMA CON AUTO_ARIMA
# ============================================================

from pmdarima import auto_arima

print("\n" + "=" * 65)
print("SELECCION AUTOMATICA DEL MODELO ARMA: AUTO_ARIMA")
print("=" * 65)

# Usar los log-retornos ya preparados en el script
serie_auto = returns["ret"].dropna().astype(float)

# Buscar modelos ARMA: d=0 porque trabajamos con log-retornos
modelo_auto = auto_arima(
    serie_auto,
    start_p=0,
    start_q=0,
    max_p=5,
    max_q=5,
    d=0,
    seasonal=False,
    information_criterion="bic",
    stepwise=True,
    with_intercept=True,
    trace=True,
    error_action="ignore",
    suppress_warnings=True
)

# Mostrar el modelo seleccionado
print("\nModelo seleccionado:")
print(f"Orden ARMA: {modelo_auto.order}")

print(f"AIC: {modelo_auto.aic():.4f}")
print(f"BIC: {modelo_auto.bic():.4f}")

print("\nResumen de estimacion:")
print(modelo_auto.summary())

# Guardar resultados
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_DIR / "resultado_auto_arima.txt",
          "w", encoding="utf-8") as archivo:
    archivo.write(f"Modelo seleccionado: ARMA{modelo_auto.order}\n")
    archivo.write(f"AIC: {modelo_auto.aic():.6f}\n")
    archivo.write(f"BIC: {modelo_auto.bic():.6f}\n\n")
    archivo.write(str(modelo_auto.summary()))

print("\nResultado guardado en Outputs/resultado_auto_arima.txt")
