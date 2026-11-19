import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

# =====================================================================
# 1. CONFIGURACIÓN DE PÁGINA (Pantalla Completa)
# =====================================================================
st.set_page_config(
    page_title="Dashboard Analítico Iris",
    page_icon="🌿",
    layout="wide"
)

# =====================================================================
# 2. CARGA DE DATASETS (Dataset 1: Morfología | Dataset 2: Geografía)
# =====================================================================
@st.cache_data
def cargar_datos():
    # Dataset 1: Medidas morfológicas (Iris)
    iris = load_iris(as_frame=True)
    df_iris = iris.frame
    df_iris.columns = ['Longitud Sépalo', 'Ancho Sépalo', 'Longitud Pétalo', 'Ancho Pétalo', 'target']
    df_iris['Especie'] = df_iris['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    
    # Dataset 2: Distribución geográfica botánica por país (Avistamientos/Cultivos)
    datos_geo = {
        'País': ['Canada', 'Russia', 'Japan', 'United States', 'United Kingdom', 'Netherlands', 'Germany', 'France', 'Mexico'],
        'ISO': ['CAN', 'RUS', 'JPN', 'USA', 'GBR', 'NLD', 'DEU', 'FRA', 'MEX'],
        'Especie': ['Setosa', 'Setosa', 'Setosa', 'Versicolor', 'Versicolor', 'Virginica', 'Virginica', 'Virginica', 'Versicolor'],
        'Registros_Botanicos': [420, 380, 290, 510, 180, 340, 210, 195, 110]
    }
    df_geo = pd.DataFrame(datos_geo)
    return df_iris, df_geo

df_iris, df_geo = cargar_datos()

# =====================================================================
# 3. BARRA LATERAL (Filtros Interactivos)
# =====================================================================
st.sidebar.header("Filtros de Análisis")
especies_sel = st.sidebar.multiselect(
    "Selecciona las especies:",
    options=df_iris['Especie'].unique(),
    default=df_iris['Especie'].unique()
)

variable_bigote = st.sidebar.selectbox(
    "Variable para Gráfica de Bigote:",
    options=['Longitud Pétalo', 'Ancho Pétalo', 'Longitud Sépalo', 'Ancho Sépalo']
)

# Filtrar ambos datasets
df_f = df_iris[df_iris['Especie'].isin(especies_sel)]
df_geo_f = df_geo[df_geo['Especie'].isin(especies_sel)]

# Validación en caso de desmarcar todas las especies
if df
