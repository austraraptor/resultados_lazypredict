# -*- coding: utf-8 -*-
"""
=====================================================================
 INFORME TÉCNICO - PREDICCIÓN DE LA DEMANDA DE INMUNIZACIONES EN ESSALUD
 Comparación de modelos con LazyPredict (regresión y clasificación)
---------------------------------------------------------------------
 Autor   : David León Callohuanca Condori
 Escuela : Ingeniería Estadística e Informática - FINESI - UNAP, Puno
 Dataset : "Inmunización EsSalud del 2015 al 2022"
           Plataforma Nacional de Datos Abiertos (datosabiertos.gob.pe)
---------------------------------------------------------------------
 CÓMO EJECUTAR
   1. Coloque este archivo .py y el CSV del dataset en la MISMA carpeta
      (el CSV debe tener "inmunizacion" en el nombre, p. ej.
      dataset_inmunizacion_2015_2023.csv). No hace falta escribir rutas.
   2. Ejecute:   python informe_lazypredict_inmunizacion.py
      (también funciona en VS Code, Spyder, Jupyter o Google Colab;
       en Colab suba el CSV a la carpeta de trabajo /content).
   3. Las librerías que falten se instalan solas en el BLOQUE 0.
   4. Todos los resultados (tablas y gráficos) se guardan en la carpeta
      "resultados_lazypredict" junto al script.
 Tiempo aproximado: 10 a 20 minutos (depende del computador).
=====================================================================
"""

# %% =================================================================
# BLOQUE 0. INSTALACIÓN AUTOMÁTICA DE LIBRERÍAS
# ====================================================================
import sys
import subprocess
import importlib.util

# (nombre para importar, nombre para pip, ¿obligatoria?)
PAQUETES = [
    ("numpy", "numpy", True),
    ("pandas", "pandas", True),
    ("scipy", "scipy", True),
    ("matplotlib", "matplotlib", True),
    ("seaborn", "seaborn", True),
    ("sklearn", "scikit-learn", True),
    ("lazypredict", "lazypredict", True),
    ("joblib", "joblib", True),
    ("openpyxl", "openpyxl", False),   # para exportar a Excel
    ("xgboost", "xgboost", False),     # modelos extra para LazyPredict
    ("lightgbm", "lightgbm", False),   # modelos extra para LazyPredict
]


def instalar(paquete_pip):
    """Instala un paquete con pip; si falla, intenta con --user."""
    base = [sys.executable, "-m", "pip", "install", "--quiet", paquete_pip]
    for extra in ([], ["--user"]):
        try:
            subprocess.check_call(base + extra)
            return True
        except Exception:
            continue
    return False


print("BLOQUE 0 - Verificando librerías...")
for modulo, pip_name, obligatoria in PAQUETES:
    if importlib.util.find_spec(modulo) is None:
        print(f"   Instalando {pip_name} ...")
        ok = instalar(pip_name)
        if not ok and obligatoria:
            raise SystemExit(f"No se pudo instalar '{pip_name}'. "
                             f"Instálelo manualmente: pip install {pip_name}")
        if not ok:
            print(f"   (opcional) {pip_name} no se instaló; se continúa sin él.")
importlib.invalidate_caches()
print("   Librerías listas.\n")

# %% =================================================================
# BLOQUE 0.1 IMPORTACIONES Y CONFIGURACIÓN GENERAL
# ====================================================================
import re
import time
import warnings
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
EN_NOTEBOOK = "ipykernel" in sys.modules
if not EN_NOTEBOOK:
    matplotlib.use("Agg")              # guarda gráficos sin abrir ventanas
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ks_2samp

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance
import joblib

from lazypredict.Supervised import LazyRegressor, LazyClassifier
from lazypredict import Supervised as LP

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEMILLA = 42
np.random.seed(SEMILLA)
sns.set_theme(style="whitegrid")

# Carpeta del script (o carpeta actual si se ejecuta en Jupyter/Colab)
try:
    CARPETA = Path(__file__).resolve().parent
except NameError:
    CARPETA = Path.cwd()
SALIDA = CARPETA / "resultados_lazypredict"
SALIDA.mkdir(exist_ok=True)


def titulo(texto):
    print("\n" + "=" * 75)
    print(" " + texto)
    print("=" * 75)


def guardar_fig(fig, nombre):
    ruta = SALIDA / f"{nombre}.png"
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    if EN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    print(f"   [gráfico guardado] {ruta.name}")


def guardar_tabla(df, nombre, index=True):
    ruta = SALIDA / f"{nombre}.csv"
    df.to_csv(ruta, index=index, encoding="utf-8-sig")
    print(f"   [tabla guardada]   {ruta.name}")


def sin_tildes(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    return texto.encode("ascii", "ignore").decode().upper().strip()


# %% =================================================================
# BLOQUE 1. PROBLEMA (breve)
# ====================================================================
titulo("1. PROBLEMA")
print("""
 Contexto : EsSalud vacuna en 403 centros de 29 redes. Cada centro debe
            prever cuántas dosis aplicará por mes y tipo de vacuna para
            planificar compras, cadena de frío y personal.
 Problema : La demanda varía según centro, red, vacuna, mes y contexto
            (COVID-19 2020-2021); un promedio histórico no basta.
 Pregunta : ¿Qué modelo de aprendizaje automático, de los que evalúa
            LazyPredict, predice mejor la cantidad mensual de personas
            vacunadas por centro asistencial y tipo de vacuna?
 Objetivo : Identificar con LazyPredict el mejor modelo de regresión
            (cantidad) y de clasificación (nivel de demanda) y validarlo
            con los registros del departamento de Puno.
""")

# %% =================================================================
# BLOQUE 2. VARIABLES (breve)
# ====================================================================
titulo("2. VARIABLES")
VARIABLES = pd.DataFrame([
    ["log_total", "Dependiente (regresión)", "Cuantitativa continua", "log(1 + total)"],
    ["nivel_demanda", "Dependiente (clasificación)", "Cualitativa ordinal", "Baja / Media / Alta (terciles)"],
    ["Año", "Independiente", "Cuantitativa discreta", "2015 a 2022"],
    ["mes", "Independiente", "Cuantitativa discreta", "1 a 12"],
    ["covid", "Independiente", "Dicotómica", "1 = 2020-2021"],
    ["red_ses", "Independiente", "Cualitativa nominal", "29 redes"],
    ["codcentro_ses", "Independiente", "Cualitativa nominal", "403 centros"],
    ["familia_vacuna", "Independiente", "Cualitativa nominal", "14 familias"],
    ["DEPARTAMENTO", "Auxiliar", "Cualitativa nominal", "Solo para separar Puno"],
], columns=["Variable", "Rol", "Tipo", "Valores"])
print(VARIABLES.to_string(index=False))

# %% =================================================================
# BLOQUE 3. INSTRUMENTO (breve)
# ====================================================================
titulo("3. TÉCNICA E INSTRUMENTO")
print("""
 Técnica      : Análisis documental de base de datos secundaria.
 Instrumentos : (1) Base de datos SES-EsSalud (CSV/XLSX)
                (2) Diccionario de datos (Formato 1-D)
                (3) Ficha de metadatos (Anexo 1-C)
                (4) Ficha de registro de datos del investigador
 Procesamiento: Python + pandas, NumPy, Matplotlib, Seaborn,
                scikit-learn y LazyPredict.
""")

# %% =================================================================
# BLOQUE 4. RECOLECCIÓN DE DATOS (carga automática del archivo)
# ====================================================================
titulo("4. RECOLECCIÓN DE DATOS")


def buscar_dataset():
    """Busca el CSV (o XLSX) del dataset en la carpeta del script y en la actual."""
    carpetas = []
    for c in (CARPETA, Path.cwd()):
        if c not in carpetas:
            carpetas.append(c)
    for patron in ("*inmunizacion*.csv", "*INMUNIZACION*.csv", "*.csv",
                   "*inmunizacion*.xlsx", "*INMUNIZACION*.xlsx"):
        for c in carpetas:
            for f in sorted(c.glob(patron)):
                nombre = f.name.lower()
                if f.parent == SALIDA or "diccionario" in nombre:
                    continue
                if f.suffix.lower() == ".csv" and patron == "*.csv":
                    # comprobar que sea el dataset correcto
                    cab = f.open(encoding="utf-8-sig", errors="ignore").readline().lower()
                    if "total" not in cab or "desclarga" not in cab:
                        continue
                return f
    raise FileNotFoundError(
        "No se encontró el dataset. Coloque el archivo CSV "
        "(p. ej. dataset_inmunizacion_2015_2023.csv) en la misma carpeta que este script.")


ARCHIVO = buscar_dataset()
print(f" Archivo encontrado: {ARCHIVO.name}")

if ARCHIVO.suffix.lower() == ".csv":
    df_raw = pd.read_csv(ARCHIVO, sep=";", encoding="utf-8-sig",
                         dtype={"UBIGEO": str}, low_memory=False)
    if df_raw.shape[1] == 1:   # por si el separador fuera coma
        df_raw = pd.read_csv(ARCHIVO, sep=",", encoding="utf-8-sig",
                             dtype={"UBIGEO": str}, low_memory=False)
    df_raw.columns = df_raw.columns.str.strip()
else:
    # Versión XLSX: tiene PERIODO (aaaamm) y no trae códigos -> se reconstruyen
    df_raw = pd.read_excel(ARCHIVO, dtype={"UBIGEO": str})
    df_raw.columns = df_raw.columns.str.strip()
    per = df_raw["PERIODO"].astype(str)
    df_raw["Año"] = per.str[:4].astype(int)
    df_raw["mes"] = per.str[4:].astype(int)
    df_raw["codcentro_ses"] = pd.factorize(df_raw["red_ses"] + "|" + df_raw["desccentro_ses"])[0]
    df_raw["codservicio"] = pd.factorize(df_raw["descservicio"])[0]

# Nombre de la columna año (puede llegar como "Año" o con codificación rota)
col_anio = [c for c in df_raw.columns if sin_tildes(c) in ("ANO", "AAO", "AO") or (c.lower().startswith("a") and len(c) <= 4)][0]
df_raw = df_raw.rename(columns={col_anio: "Año"})
df_raw["UBIGEO"] = df_raw["UBIGEO"].astype(str).str.zfill(6)

print(f" Registros: {df_raw.shape[0]:,}   Variables: {df_raw.shape[1]}")
print(f" Periodo  : {df_raw['Año'].min()}-{df_raw['Año'].max()}   "
      f"Dosis totales: {df_raw['total'].sum():,}")
print("\n Primeras filas:")
print(df_raw.head(3).to_string())

# %% =================================================================
# BLOQUE 5. TABULACIÓN
# ====================================================================
titulo("5. TABULACIÓN")
print(" Los datos ya vienen tabulados por la fuente (matriz en formato largo:")
print(" 1 fila = centro x mes x vacuna). Se presentan tablas de frecuencia.\n")

tab_anio = df_raw.groupby("Año").agg(Registros=("total", "size"), Dosis=("total", "sum"))
tab_anio["% Dosis"] = (tab_anio["Dosis"] / tab_anio["Dosis"].sum() * 100).round(2)
print(" Tabla - Registros y dosis por año\n", tab_anio.to_string())
guardar_tabla(tab_anio, "T01_por_anio")

tab_mes = df_raw.groupby("mes").agg(Dosis=("total", "sum"))
tab_mes["Promedio anual"] = (tab_mes["Dosis"] / df_raw["Año"].nunique()).round(0)
tab_mes["% Dosis"] = (tab_mes["Dosis"] / tab_mes["Dosis"].sum() * 100).round(2)
print("\n Tabla - Dosis por mes\n", tab_mes.to_string())
guardar_tabla(tab_mes, "T02_por_mes")

# %% =================================================================
# BLOQUE 6. LIMPIEZA Y DIAGNÓSTICO DE DATOS FALTANTES (MCAR, MAR, MNAR)
# ====================================================================
titulo("6. LIMPIEZA Y DIAGNÓSTICO DE FALTANTES (MCAR / MAR / MNAR)")
df = df_raw.copy()

# 6.1 Faltantes explícitos (NaN, vacíos y textos de relleno)
print("\n 6.1 Faltantes explícitos por variable")
texto_vacio = {"", "-", "NA", "N/A", "NULL", "NAN", "SIN DATO", "S/D"}
obj_cols = df.select_dtypes(include=["object", "string"]).columns
vacios = {c: int(df[c].astype(str).str.strip().str.upper().isin(texto_vacio).sum()) for c in obj_cols}
falt = pd.DataFrame({"NaN": df.isna().sum(),
                     "Vacíos/relleno": pd.Series(vacios)}).fillna(0).astype(int)
falt["% faltante"] = ((falt["NaN"] + falt["Vacíos/relleno"]) / len(df) * 100).round(4)
print(falt.to_string())
guardar_tabla(falt, "T03_faltantes_explicitos")
total_falt = int(falt["NaN"].sum() + falt["Vacíos/relleno"].sum())
print(f"\n   Total de celdas faltantes explícitas: {total_falt}")
if total_falt == 0:
    print("   -> No hay faltantes explícitos. La prueba de Little (MCAR) no aplica,")
    print("      porque necesita valores perdidos en al menos una variable.")

# 6.2 Faltantes implícitos: meses en que un centro no reporta ninguna vacuna
print("\n 6.2 Faltantes implícitos (panel centro x mes)")
df["periodo"] = df["Año"] * 100 + df["mes"]
periodos = np.sort(df["periodo"].unique())
idx_per = {p: i for i, p in enumerate(periodos)}
obs = df[["codcentro_ses", "periodo"]].drop_duplicates()
obs["i"] = obs["periodo"].map(idx_per)
rango = obs.groupby("codcentro_ses")["i"].agg(["min", "max"])
filas = []
for centro, r in rango.iterrows():
    for i in range(int(r["min"]), int(r["max"]) + 1):
        filas.append((centro, periodos[i]))
panel = pd.DataFrame(filas, columns=["codcentro_ses", "periodo"])
panel = panel.merge(obs[["codcentro_ses", "periodo"]].assign(reporta=1),
                    on=["codcentro_ses", "periodo"], how="left")
panel["faltante"] = panel["reporta"].isna().astype(int)
panel["Año"] = panel["periodo"] // 100
red_map = df.drop_duplicates("codcentro_ses").set_index("codcentro_ses")["red_ses"]
panel["red_ses"] = panel["codcentro_ses"].map(red_map)
n_falt = int(panel["faltante"].sum())
print(f"   Centro-mes esperados (entre 1.er y último reporte): {len(panel):,}")
print(f"   Centro-mes sin ningún reporte                     : {n_falt:,} "
      f"({n_falt / len(panel) * 100:.2f} %)")

tasa_anio = (panel.groupby("Año")["faltante"].mean() * 100).round(2)
print("\n   % centro-mes faltantes por año:\n", tasa_anio.to_string())
guardar_tabla(tasa_anio.to_frame("% faltante"), "T04_faltantes_implicitos_anio")

# Pruebas: si la ausencia depende de variables observadas -> no es MCAR (es MAR)
chi_a, p_a, _, _ = chi2_contingency(pd.crosstab(panel["Año"], panel["faltante"]))
chi_r, p_r, _, _ = chi2_contingency(pd.crosstab(panel["red_ses"], panel["faltante"]))
print(f"\n   Chi-cuadrado faltante vs Año : chi2 = {chi_a:,.2f}  p = {p_a:.4g}")
print(f"   Chi-cuadrado faltante vs Red : chi2 = {chi_r:,.2f}  p = {p_r:.4g}")
mecanismo = "MAR" if (p_a < 0.05 or p_r < 0.05) else "MCAR"
print(f"""
   INTERPRETACIÓN
   - MCAR (completamente al azar): se rechaza si p < 0.05. Resultado: {'se rechaza' if mecanismo == 'MAR' else 'no se rechaza'}.
   - MAR (al azar condicionado): la ausencia depende de variables observadas
     (año y/o red). Mecanismo más compatible con los datos: {mecanismo}.
   - MNAR (no al azar): un centro que no vacunó en un mes probablemente no
     reporta (el valor ausente sería 0). Esto no se puede probar con los datos
     observados; se reconoce como posible y se documenta como limitación.""")

# 6.3 Operaciones de limpieza
print("\n 6.3 Operaciones de limpieza")
n0 = len(df)
dups = int(df.duplicated().sum())
df = df.drop_duplicates()
print(f"   Duplicados eliminados: {dups}")
clave = ["Año", "mes", "codcentro_ses", "codservicio"]
print(f"   Repeticiones de la clave {clave}: {int(df.duplicated(clave).sum())}")
fuera = int((df["total"] <= 0).sum())
df = df[df["total"] > 0]
print(f"   Registros con total <= 0 eliminados: {fuera}")

for c in df.select_dtypes(include=["object", "string"]).columns:
    df[c] = df[c].astype(str).str.strip()

# Normalización de departamentos (31 etiquetas -> 25 departamentos)
antes_dep = df_raw["DEPARTAMENTO"].nunique()
CORR_DEP = {"LALIBERTAD": "LA LIBERTAD", "SANMARTIN": "SAN MARTIN",
            "MADREDEDIOS": "MADRE DE DIOS", "HUANTA": "AYACUCHO"}
df["DEPARTAMENTO"] = df["DEPARTAMENTO"].map(sin_tildes).replace(CORR_DEP)
print(f"   Departamentos: {antes_dep} etiquetas -> {df['DEPARTAMENTO'].nunique()} departamentos")

# UBIGEO con código de departamento fuera del catálogo INEI (01-25)
cod_dep = pd.to_numeric(df["UBIGEO"].str[:2], errors="coerce")
df["ubigeo_valido"] = cod_dep.between(1, 25).astype(int)
print(f"   UBIGEO fuera del catálogo INEI: {int((df['ubigeo_valido'] == 0).sum()):,} registros "
      f"(no afecta: UBIGEO no es predictora)")

# Normalización de nombres de vacuna (tildes / mayúsculas)
antes_vac = df["desclarga"].nunique()
df["vacuna_norm"] = df["desclarga"].map(sin_tildes)
print(f"   Nombres de vacuna: {antes_vac} -> {df['vacuna_norm'].nunique()} al quitar tildes/mayúsculas")
print(f"   Registros finales: {len(df):,} de {n0:,}")

# %% =================================================================
# BLOQUE 7. IMPUTACIÓN
# ====================================================================
titulo("7. IMPUTACIÓN")
print(f"""
 Decisión:
 - Faltantes explícitos = {total_falt} -> NO se requiere imputación.
 - Faltantes implícitos ({n_falt:,} centro-mes) -> NO se imputan: la unidad
   de análisis es el registro reportado; imputarlos inventaría atenciones
   que no ocurrieron o no se registraron (posible MNAR).
 - Como resguardo, LazyPredict incluye imputación interna en su pipeline:
   media para numéricas y la categoría "missing" para categóricas.
""")
quedan = int(df[["Año", "mes", "red_ses", "codcentro_ses", "desclarga", "total"]].isna().sum().sum())
print(f" Verificación: valores faltantes en las variables del modelo = {quedan}")

# %% =================================================================
# BLOQUE 7.1 CONSTRUCCIÓN DE VARIABLES (familia de vacuna y COVID)
# ====================================================================
titulo("7.1 CONSTRUCCIÓN DE VARIABLES")
REGLAS_FAMILIA = [   # el orden importa: Hib va antes que Influenza
    ("Hib", r"HAEMOPHILUS"),
    ("Influenza", r"INFLUENZA"),
    ("Pentavalente", r"PENTAVALENTE"),
    ("DPT", r"\bDPT\b"),
    ("Difteria-Tetano", r"DIFTERIA|DIFETERIA|DIFTOTETANO|\bDT\b|DTPA"),
    ("Poliomielitis", r"POLI"),
    ("Neumococo", r"NEUMOCOC"),
    ("Hepatitis B", r"HEPATITIS|ANTIH\.?B"),
    ("Sarampion-Rubeola-Paperas", r"SARAMPI|ANTISARAMPIONOSA|\(SR\)"),
    ("Fiebre amarilla", r"AMARIL"),
    ("Rotavirus", r"ROTAVIRUS"),
    ("BCG", r"BCG"),
    ("Varicela", r"VARICELA"),
    ("VPH", r"PAPILOMA"),
]


def familia_vacuna(nombre):
    for fam, patron in REGLAS_FAMILIA:
        if re.search(patron, nombre):
            return fam
    return "Otra"


df["familia_vacuna"] = df["vacuna_norm"].map(familia_vacuna)
df["covid"] = df["Año"].isin([2020, 2021]).astype(int)
df["codcentro_ses"] = df["codcentro_ses"].astype(str)
df["red_ses"] = df["red_ses"].astype(str)

tab_fam = df.groupby("familia_vacuna").agg(Nombres=("desclarga", "nunique"),
                                           Registros=("total", "size"),
                                           Dosis=("total", "sum")).sort_values("Dosis", ascending=False)
tab_fam["% Dosis"] = (tab_fam["Dosis"] / tab_fam["Dosis"].sum() * 100).round(2)
print(tab_fam.to_string())
guardar_tabla(tab_fam, "T05_familias_vacuna")
print(f"\n Familias: {df['familia_vacuna'].nunique()}   "
      f"Registros sin familia asignada: {int((df['familia_vacuna'] == 'Otra').sum())}")

# %% =================================================================
# BLOQUE 8. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ====================================================================
titulo("8. ANÁLISIS EXPLORATORIO DE DATOS (EDA)")

# 8.1 Estadística descriptiva de total
t = df["total"]
q1, q3 = t.quantile([0.25, 0.75])
desc = pd.Series({
    "Media": t.mean(), "Mediana": t.median(), "Desv. estándar": t.std(),
    "Coef. variación": t.std() / t.mean(), "Mínimo": t.min(), "Máximo": t.max(),
    "Asimetría": t.skew(), "Curtosis": t.kurt(),
    "Asimetría log(1+total)": np.log1p(t).skew(),
    "% atípicos (IQR)": (t > q3 + 1.5 * (q3 - q1)).mean() * 100,
}).round(3)
print(" 8.1 Descriptivos de 'total'\n", desc.to_string())
guardar_tabla(desc.to_frame("Valor"), "T06_descriptivos_total")

# 8.2 Distribución: original vs logarítmica
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(t[t <= t.quantile(0.99)], bins=60, color="#4C72B0")
ax[0].set_title("Distribución de total (hasta p99)")
ax[0].set_xlabel("Personas vacunadas"); ax[0].set_ylabel("Frecuencia")
ax[1].hist(np.log1p(t), bins=60, color="#55A868")
ax[1].set_title("Distribución de log(1 + total)")
ax[1].set_xlabel("log(1 + total)")
guardar_fig(fig, "G01_distribucion_total")

# 8.3 Serie anual y estacionalidad
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
anual = df.groupby("Año")["total"].sum() / 1e6
ax[0].plot(anual.index, anual.values, marker="o", color="#4C72B0")
ax[0].axvspan(2019.5, 2021.5, color="#C44E52", alpha=0.15, label="COVID-19")
ax[0].set_title("Dosis aplicadas por año"); ax[0].set_ylabel("Millones de dosis")
ax[0].legend()
mensual = df.groupby("mes")["total"].sum() / df["Año"].nunique() / 1e3
ax[1].bar(mensual.index, mensual.values, color="#4C72B0")
ax[1].set_title("Promedio de dosis por mes"); ax[1].set_ylabel("Miles de dosis")
ax[1].set_xticks(range(1, 13))
guardar_fig(fig, "G02_anual_y_estacionalidad")

# 8.4 Mapa de calor año x mes
calor = df.pivot_table(index="Año", columns="mes", values="total", aggfunc="sum") / 1e3
fig, ax = plt.subplots(figsize=(11, 4.5))
sns.heatmap(calor, cmap="Blues", annot=True, fmt=".0f", cbar_kws={"label": "Miles de dosis"}, ax=ax)
ax.set_title("Dosis por año y mes (miles)")
guardar_fig(fig, "G03_calor_anio_mes")

# 8.5 Familias de vacuna y redes
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fam = df.groupby("familia_vacuna")["total"].sum().sort_values() / 1e6
ax[0].barh(fam.index, fam.values, color="#4C72B0")
ax[0].set_title("Dosis por familia de vacuna"); ax[0].set_xlabel("Millones de dosis")
red = (df.groupby("red_ses")["total"].sum().sort_values().tail(12)) / 1e6
ax[1].barh(red.index, red.values, color="#8172B2")
ax[1].set_title("Top 12 redes por dosis"); ax[1].set_xlabel("Millones de dosis")
guardar_fig(fig, "G04_familias_y_redes")

# 8.6 Boxplot de log_total por familia
fig, ax = plt.subplots(figsize=(12, 5))
orden = df.groupby("familia_vacuna")["total"].median().sort_values().index
sns.boxplot(data=df.assign(log_total=np.log1p(df["total"])), x="familia_vacuna",
            y="log_total", order=orden, color="#A1C9F4", fliersize=1, ax=ax)
ax.set_title("log(1 + total) por familia de vacuna"); ax.set_xlabel("")
ax.tick_params(axis="x", rotation=45)
guardar_fig(fig, "G05_boxplot_familias")

# 8.7 Puno frente al resto del país
df["ambito"] = np.where(df["DEPARTAMENTO"] == "PUNO", "Puno", "Resto del país")
comp = df.groupby("ambito")["total"].agg(Registros="size", Dosis="sum", Media="mean", Mediana="median").round(2)
print("\n 8.7 Puno vs resto del país\n", comp.to_string())
guardar_tabla(comp, "T07_puno_vs_resto")
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df.assign(log_total=np.log1p(df["total"])), x="ambito", y="log_total",
            hue="ambito", palette=["#C44E52", "#4C72B0"], legend=False, fliersize=1, ax=ax)
ax.set_title("log(1 + total): Puno vs resto del país"); ax.set_xlabel("")
guardar_fig(fig, "G06_puno_vs_resto")

# 8.8 Asociación de cada predictora con log_total (eta cuadrado)
def eta2(y, g):
    m = y.mean()
    ssb = (y.groupby(g).agg(["count", "mean"]).pipe(lambda d: (d["count"] * (d["mean"] - m) ** 2).sum()))
    return ssb / ((y - m) ** 2).sum()

ylog = np.log1p(df["total"])
asoc = pd.Series({c: eta2(ylog, df[c]) for c in
                  ["Año", "mes", "covid", "red_ses", "codcentro_ses", "familia_vacuna"]}
                 ).sort_values(ascending=False).round(4)
print("\n 8.8 Eta² (proporción de la varianza de log_total explicada por cada variable)\n",
      asoc.to_string())
guardar_tabla(asoc.to_frame("Eta2"), "T08_eta2_predictoras")
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(asoc.index[::-1], asoc.values[::-1], color="#55A868")
ax.set_title("Asociación con log(1 + total) - Eta²"); ax.set_xlabel("Eta²")
guardar_fig(fig, "G07_eta2_predictoras")

# %% =================================================================
# BLOQUE 9. VARIABLE OBJETIVO (TARGET)
# ====================================================================
titulo("9. VARIABLE OBJETIVO (TARGET)")
df["log_total"] = np.log1p(df["total"])
c1, c2 = df["total"].quantile([1 / 3, 2 / 3])
c1, c2 = int(np.floor(c1)), int(np.floor(c2))
df["nivel_demanda"] = pd.cut(df["total"], bins=[0, c1, c2, np.inf],
                             labels=["Baja", "Media", "Alta"]).astype(str)
print(f" Regresión     -> y = log_total = log(1 + total)   "
      f"(asimetría {df['total'].skew():.2f} -> {df['log_total'].skew():.2f})")
print(f" Clasificación -> y = nivel_demanda: Baja (<= {c1}), Media ({c1 + 1}-{c2}), Alta (>= {c2 + 1})")
dist_cls = df["nivel_demanda"].value_counts().reindex(["Baja", "Media", "Alta"])
dist_cls = pd.DataFrame({"Registros": dist_cls, "%": (dist_cls / len(df) * 100).round(2)})
print(dist_cls.to_string())
guardar_tabla(dist_cls, "T09_distribucion_nivel_demanda")
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(dist_cls.index, dist_cls["Registros"], color=["#8DA0CB", "#66C2A5", "#FC8D62"])
ax.set_title("Clases de nivel_demanda (balanceadas)"); ax.set_ylabel("Registros")
guardar_fig(fig, "G08_clases_nivel_demanda")

# %% =================================================================
# BLOQUE 10. FORMULACIÓN DEL MODELO (taxonomía de modelos)
# ====================================================================
titulo("10. FORMULACIÓN DEL MODELO")
print("""
 Dimensión        Pregunta                                   Decisión
 ---------------  -----------------------------------------  -------------------------------
 Representación   ¿Cómo se representa el sistema?            Modelo matemático-estadístico
                                                             (datos tabulares supervisados)
 Incertidumbre    ¿Los eventos son conocidos con certeza?    Estocástico (la demanda varía)
 Tiempo           ¿La decisión es en una etapa o evoluciona? Estático: predicción por registro
                                                             con año y mes como predictoras
 Propósito        ¿Entender, anticipar o recomendar?         ANTICIPAR (modelo predictivo)

 Métodos analíticos aplicados: REGRESIÓN (cuántas dosis) y CLASIFICACIÓN (qué nivel).
 No aplicados: series temporales (posible extensión), clustering, reglas de
 asociación y detección de anomalías (no responden a la pregunta planteada).
 Arquitectura de datos: los datos se actualizan por lotes mensuales, por lo que
 encaja en la capa batch de una arquitectura LAMBDA (no hay flujo en tiempo real
 que justifique KAPPA).
""")

# %% =================================================================
# BLOQUE 11. MUESTREO Y PARTICIÓN DE DATOS
# ====================================================================
titulo("11. MUESTREO Y PARTICIÓN")
N_MUESTRA = 30000
frac = N_MUESTRA / len(df)
muestra = df.groupby(["Año", "familia_vacuna"]).sample(frac=frac, random_state=SEMILLA)
puno_val = df[(df["DEPARTAMENTO"] == "PUNO") & (~df.index.isin(muestra.index))].copy()
ks = ks_2samp(muestra["log_total"], df["log_total"])
print(f" Muestra estratificada (Año x familia): {len(muestra):,} registros ({frac * 100:.2f} %)")
print(f" Prueba K-S muestra vs población (log_total): D = {ks.statistic:.4f}, p = {ks.pvalue:.4f}")
print(f" Validación local Puno (fuera de la muestra): {len(puno_val):,} registros")

PREDICTORAS = ["Año", "mes", "covid", "red_ses", "codcentro_ses", "familia_vacuna"]
X = muestra[PREDICTORAS].copy()
y_reg = muestra["log_total"].copy()
y_clf = muestra["nivel_demanda"].copy()

X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_clf, test_size=0.20, random_state=SEMILLA, stratify=y_clf)
print(f" Entrenamiento: {len(X_train):,}   Prueba: {len(X_test):,}")


def codificar_onehot(Xd):
    """familia_vacuna (14) y red_ses (29) en one-hot; codcentro_ses queda para
    la codificación ordinal automática de LazyPredict (403 categorías)."""
    Xd = pd.get_dummies(Xd, columns=["familia_vacuna", "red_ses"], dtype=int)
    return Xd


Xtr_lp = codificar_onehot(X_train)
Xte_lp = codificar_onehot(X_test).reindex(columns=Xtr_lp.columns, fill_value=0)
print(f" Columnas tras la codificación: {Xtr_lp.shape[1]}")

# Modelos excluidos por costo computacional O(n²) o O(n³) con ~24 000 filas
EXCLUIR_REG = {"KernelRidge", "GaussianProcessRegressor", "SVR", "NuSVR", "QuantileRegressor"}
EXCLUIR_CLF = {"SVC", "NuSVC", "LabelPropagation", "LabelSpreading", "GaussianProcessClassifier"}
lista_reg = [cls for nombre, cls in LP.REGRESSORS if nombre not in EXCLUIR_REG]
lista_clf = [cls for nombre, cls in LP.CLASSIFIERS if nombre not in EXCLUIR_CLF]
print(f" Modelos a evaluar: {len(lista_reg)} regresores y {len(lista_clf)} clasificadores")
print(f" Excluidos por costo computacional: {sorted(EXCLUIR_REG | EXCLUIR_CLF)}")


def resultados(salida_fit):
    """LazyPredict devuelve una tabla o (tabla, predicciones) según la versión."""
    return salida_fit[0] if isinstance(salida_fit, tuple) else salida_fit


# %% =================================================================
# BLOQUE 12. LAZYPREDICT - REGRESIÓN (LazyRegressor)
# ====================================================================
titulo("12. LAZYREGRESSOR (y = log_total)")
t0 = time.time()
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None,
                    regressors=lista_reg, random_state=SEMILLA)
res_reg = resultados(reg.fit(Xtr_lp, Xte_lp, yr_train, yr_test))
res_reg = res_reg.sort_values("Adjusted R-Squared", ascending=False)
vista = res_reg.round(4).astype(object).mask(res_reg.abs() > 1e6, "divergió")
print(vista.to_string())
print(" ('divergió' = el algoritmo no convergió con los parámetros por defecto)")
print(f"\n Tiempo total LazyRegressor: {time.time() - t0:.1f} s")
guardar_tabla(res_reg.round(4), "T10_lazyregressor_resultados")

top = res_reg.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top.index, top["R-Squared"].clip(lower=0), color="#4C72B0")
ax.set_title("Top 10 modelos de regresión - R²"); ax.set_xlabel("R² (conjunto de prueba)")
guardar_fig(fig, "G09_top10_regresion")

# %% =================================================================
# BLOQUE 13. LAZYPREDICT - CLASIFICACIÓN (LazyClassifier)
# ====================================================================
titulo("13. LAZYCLASSIFIER (y = nivel_demanda)")
t0 = time.time()
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None,
                     classifiers=lista_clf, random_state=SEMILLA)
res_clf = resultados(clf.fit(Xtr_lp, Xte_lp, yc_train, yc_test))
res_clf = res_clf.sort_values("Balanced Accuracy", ascending=False)
print(res_clf.round(4).to_string())
print(f"\n Tiempo total LazyClassifier: {time.time() - t0:.1f} s")
guardar_tabla(res_clf.round(4), "T11_lazyclassifier_resultados")

top = res_clf.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top.index, top["Balanced Accuracy"], color="#DD8452")
ax.set_title("Top 10 modelos de clasificación - Balanced Accuracy")
ax.set_xlabel("Balanced Accuracy (conjunto de prueba)")
guardar_fig(fig, "G10_top10_clasificacion")

# %% =================================================================
# BLOQUE 14. SELECCIÓN DEL MODELO (contraste de métricas)
# ====================================================================
titulo("14. SELECCIÓN DEL MODELO")
validos_reg = res_reg[res_reg["R-Squared"].notna()]
validos_clf = res_clf[res_clf["Balanced Accuracy"].notna()]
mejor_reg = validos_reg.index[0]
mejor_clf = validos_clf.index[0]
print(" Top 5 regresión (criterio: R² ajustado; desempate: RMSE y tiempo)")
print(validos_reg.head(5)[["Adjusted R-Squared", "R-Squared", "RMSE", "Time Taken"]].round(4).to_string())
print("\n Top 5 clasificación (criterio: Balanced Accuracy; desempate: F1 y tiempo)")
cols_c = [c for c in ["Accuracy", "Balanced Accuracy", "F1 Score", "Time Taken"] if c in validos_clf.columns]
print(validos_clf.head(5)[cols_c].round(4).to_string())
print(f"\n >>> Mejor regresor   : {mejor_reg}")
print(f" >>> Mejor clasificador: {mejor_clf}")

# %% =================================================================
# BLOQUE 15. VALIDACIÓN DEL MEJOR MODELO (prueba + Puno)
# ====================================================================
titulo("15. VALIDACIÓN DEL MEJOR MODELO")
dict_reg = dict(LP.REGRESSORS)
dict_clf = dict(LP.CLASSIFIERS)


def crear_modelo(clase):
    extra = {"verbose": -1} if "LGBM" in clase.__name__ else {}
    try:
        return clase(random_state=SEMILLA, **extra)
    except TypeError:
        return clase(**extra)


def preprocesador(Xd):
    """Replica el preprocesamiento de LazyPredict: numéricas -> media + escala;
    categóricas de alta cardinalidad -> codificación ordinal."""
    num = Xd.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in Xd.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")),
                          ("esc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                 unknown_value=-1))]), cat),
    ])


Xpuno = codificar_onehot(puno_val[PREDICTORAS]).reindex(columns=Xtr_lp.columns, fill_value=0)

# 15.1 Regresión
pipe_r = Pipeline([("prep", preprocesador(Xtr_lp)), ("modelo", crear_modelo(dict_reg[mejor_reg]))])
pipe_r.fit(Xtr_lp, yr_train)


def metricas_reg(y_log, pred_log):
    y_real, y_pred = np.expm1(y_log), np.expm1(pred_log)
    return {"R² (log)": r2_score(y_log, pred_log),
            "RMSE (log)": np.sqrt(mean_squared_error(y_log, pred_log)),
            "MAE (personas)": mean_absolute_error(y_real, y_pred),
            "Mediana error abs. (personas)": np.median(np.abs(y_real - y_pred))}


val_reg = pd.DataFrame({
    "Prueba nacional": metricas_reg(yr_test, pipe_r.predict(Xte_lp)),
    "Validación Puno": metricas_reg(puno_val["log_total"], pipe_r.predict(Xpuno)),
}).round(4)
print(f" Regresión - {mejor_reg}\n", val_reg.to_string())
guardar_tabla(val_reg, "T12_validacion_regresion")

pred_test = pipe_r.predict(Xte_lp)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(yr_test, pred_test, s=4, alpha=0.3, color="#4C72B0")
lim = [yr_test.min(), yr_test.max()]
ax.plot(lim, lim, color="#C44E52", lw=1.5)
ax.set_title(f"Real vs predicho - {mejor_reg}")
ax.set_xlabel("log_total real"); ax.set_ylabel("log_total predicho")
guardar_fig(fig, "G11_real_vs_predicho")

# 15.2 Clasificación
pipe_c = Pipeline([("prep", preprocesador(Xtr_lp)), ("modelo", crear_modelo(dict_clf[mejor_clf]))])
pipe_c.fit(Xtr_lp, yc_train)


def metricas_clf(y, p):
    return {"Accuracy": accuracy_score(y, p),
            "Balanced Accuracy": balanced_accuracy_score(y, p),
            "F1 ponderado": f1_score(y, p, average="weighted")}


pc_test = pipe_c.predict(Xte_lp)
pc_puno = pipe_c.predict(Xpuno)
val_clf = pd.DataFrame({"Prueba nacional": metricas_clf(yc_test, pc_test),
                        "Validación Puno": metricas_clf(puno_val["nivel_demanda"], pc_puno)}).round(4)
print(f"\n Clasificación - {mejor_clf}\n", val_clf.to_string())
guardar_tabla(val_clf, "T13_validacion_clasificacion")
print("\n Reporte por clase (prueba nacional):")
print(classification_report(yc_test, pc_test, digits=3))

etiquetas = ["Baja", "Media", "Alta"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, (yy, pp, nom) in zip(ax, [(yc_test, pc_test, "Prueba nacional"),
                                 (puno_val["nivel_demanda"], pc_puno, "Validación Puno")]):
    cm = confusion_matrix(yy, pp, labels=etiquetas)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=etiquetas,
                yticklabels=etiquetas, ax=a)
    a.set_title(f"{nom}"); a.set_xlabel("Predicho"); a.set_ylabel("Real")
fig.suptitle(f"Matrices de confusión - {mejor_clf}")
guardar_fig(fig, "G12_matrices_confusion")

# 15.3 Importancia de variables (permutación, sobre el regresor)
sub = Xte_lp.sample(min(3000, len(Xte_lp)), random_state=SEMILLA)
imp = permutation_importance(pipe_r, sub, yr_test.loc[sub.index], n_repeats=5,
                             random_state=SEMILLA, scoring="r2")
imp_col = pd.Series(imp.importances_mean, index=sub.columns)
grupo = imp_col.groupby(lambda c: "familia_vacuna" if c.startswith("familia_vacuna_")
                        else ("red_ses" if c.startswith("red_ses_") else c)).sum()
grupo = grupo.sort_values(ascending=False).round(4)
print(f"\n Importancia por permutación (caída de R²) - {mejor_reg}\n", grupo.to_string())
guardar_tabla(grupo.to_frame("Caída de R²"), "T14_importancia_variables")
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(grupo.index[::-1], grupo.values[::-1], color="#55A868")
ax.set_title(f"Importancia de variables - {mejor_reg}"); ax.set_xlabel("Caída de R²")
guardar_fig(fig, "G13_importancia_variables")

# %% =================================================================
# BLOQUE 16. IMPLEMENTACIÓN (guardar modelos y resumen)
# ====================================================================
titulo("16. IMPLEMENTACIÓN Y RESUMEN")
joblib.dump(pipe_r, SALIDA / "modelo_regresion.joblib")
joblib.dump(pipe_c, SALIDA / "modelo_clasificacion.joblib")
print("   [modelos guardados] modelo_regresion.joblib, modelo_clasificacion.joblib")

try:
    with pd.ExcelWriter(SALIDA / "resultados_lazypredict.xlsx") as xls:
        res_reg.round(4).to_excel(xls, sheet_name="LazyRegressor")
        res_clf.round(4).to_excel(xls, sheet_name="LazyClassifier")
        val_reg.to_excel(xls, sheet_name="Validacion_regresion")
        val_clf.to_excel(xls, sheet_name="Validacion_clasificacion")
        grupo.to_frame("Caida_R2").to_excel(xls, sheet_name="Importancia")
    print("   [excel guardado]   resultados_lazypredict.xlsx")
except Exception as e:
    print(f"   (No se generó el Excel: {e}. Los CSV sí están disponibles.)")

resumen = f"""
RESUMEN DEL ANÁLISIS
--------------------
Dataset              : {ARCHIVO.name} ({len(df):,} registros, {df['Año'].min()}-{df['Año'].max()})
Faltantes explícitos : {total_falt}  | Faltantes implícitos centro-mes: {n_falt:,} (mecanismo: {mecanismo})
Muestra LazyPredict  : {len(muestra):,} (K-S p = {ks.pvalue:.3f}) | Validación Puno: {len(puno_val):,}

REGRESIÓN (y = log_total)
  Mejor modelo       : {mejor_reg}
  R² ajustado (Lazy) : {validos_reg.iloc[0]['Adjusted R-Squared']:.4f}
  R² prueba / Puno   : {val_reg.loc['R² (log)', 'Prueba nacional']:.4f} / {val_reg.loc['R² (log)', 'Validación Puno']:.4f}
  MAE prueba / Puno  : {val_reg.loc['MAE (personas)', 'Prueba nacional']:.2f} / {val_reg.loc['MAE (personas)', 'Validación Puno']:.2f} personas

CLASIFICACIÓN (y = nivel_demanda)
  Mejor modelo       : {mejor_clf}
  Bal. Acc. (Lazy)   : {validos_clf.iloc[0]['Balanced Accuracy']:.4f}
  Bal. Acc. prueba / Puno: {val_clf.loc['Balanced Accuracy', 'Prueba nacional']:.4f} / {val_clf.loc['Balanced Accuracy', 'Validación Puno']:.4f}

Variable más importante: {grupo.index[0]}
"""
print(resumen)
(SALIDA / "resumen.txt").write_text(resumen, encoding="utf-8")
print(f" Todos los resultados están en: {SALIDA}")
