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
if df_f.empty:
    st.warning("Por favor, selecciona al menos una especie en el menú lateral para visualizar los datos.")
    st.stop()

# =====================================================================
# 4. TÍTULO RESALTADO Y PREGUNTA DE INVESTIGACIÓN
# =====================================================================
st.markdown(
    """
    <div style='background-color:#1B5E20; padding:20px; border-radius:10px; margin-bottom:20px;'>
        <h1 style='color:white; text-align:center; margin:0;'>🌿 Monitor Analítico de Morfología y Distribución Global del Género Iris</h1>
        <p style='color:#E8F5E9; text-align:center; font-size:18px; margin-top:10px;'>
            <b>Pregunta de Investigación:</b> ¿De qué manera se correlacionan las dimensiones morfológicas de las especies del género Iris y cómo se distribuye su presencia geográfica a nivel internacional?
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================================
# 5. FILA 1: KPIs 1, 2 y 3 (Tarjetas de Métricas Clave)
# =====================================================================
k1, k2, k3 = st.columns(3)
with k1:
    st.metric(label="KPI 1: Total de Especímenes", value=f"{len(df_f)} flores")
    st.caption("Volumen activo de la muestra según los filtros seleccionados.")
with k2:
    st.metric(label="KPI 2: Longitud Promedio Pétalo", value=f"{df_f['Longitud Pétalo'].mean():.2f} cm")
    st.caption("Medida principal: varía drásticamente entre especies.")
with k3:
    st.metric(label="KPI 3: Longitud Promedio Sépalo", value=f"{df_f['Longitud Sépalo'].mean():.2f} cm")
    st.caption("Medida de contraste: presenta menor variación entre especies.")

st.divider()

# =====================================================================
# 6. FILA 2: KPI 4 (Mapa Geográfico) y KPI 5 (Regresión Lineal)
# =====================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("KPI 4: Mapa Geográfico de Distribución")
    fig_mapa = px.choropleth(
        df_geo_f,
        locations="ISO",
        color="Registros_Botanicos",
        hover_name="País",
        hover_data=["Especie"],
        color_continuous_scale=px.colors.sequential.Greens,
        title="Registros Botánicos Globales por País"
    )
    fig_mapa.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_mapa, use_container_width=True)
    st.caption("Interpretación: Los tonos más oscuros indican mayor concentración de registros botánicos por país.")

with col2:
    st.subheader("KPI 5: Regresión Lineal (Pétalo)")
    fig_reg = px.scatter(
        df_f,
        x="Longitud Pétalo",
        y="Ancho Pétalo",
        color="Especie",
        trendline="ols",
        title="Tendencia Lineal: Longitud vs. Ancho del Pétalo (cm)"
    )
    st.plotly_chart(fig_reg, use_container_width=True)
    st.caption("Interpretación: Relación directa; a mayor longitud del pétalo, mayor es su ancho.")

# =====================================================================
# 7. FILA 3: KPI 6 (Mapa de Calor) y KPI 7 (Gráfica de Bigote)
# =====================================================================
col3, col4 = st.columns(2)

with col3:
    st.subheader("KPI 6: Mapa de Calor (Correlación)")
    corr = df_f[['Longitud Sépalo', 'Ancho Sépalo', 'Longitud Pétalo', 'Ancho Pétalo']].corr()
    fig_heat = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Matriz de Correlación entre Variables"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Interpretación: Valores cercanos a 1 (verde intenso) confirman que las medidas crecen juntas.")

with col4:
    st.subheader(f"KPI 7: Gráfica de Bigote ({variable_bigote})")
    fig_box = px.box(
        df_f,
        x="Especie",
        y=variable_bigote,
        color="Especie",
        points="all",
        title=f"Distribución y Valores Atípicos de {variable_bigote}"
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("Interpretación: Cajas compactas indican tamaños uniformes; puntos fuera de los bigotes son valores atípicos.")

st.divider()

# =====================================================================
# 8. FILA 4: KPI 8 (Gráfico de Dona) y KPI 9 (Tabla Resumen)
# =====================================================================
col5, col6 = st.columns(2)

with col5:
    st.subheader("KPI 8: Proporción de Muestras")
    conteo = df_f['Especie'].value_counts().reset_index()
    conteo.columns = ['Especie', 'Cantidad']
    fig_dona = px.pie(
        conteo,
        values='Cantidad',
        names='Especie',
        hole=0.5,
        title="Distribución Porcentual del Dataset"
    )
    st.plotly_chart(fig_dona, use_container_width=True)
    st.caption("Interpretación: Confirma el equilibrio de la muestra sin sesgos hacia ninguna especie.")

with col6:
    st.subheader("KPI 9: Resumen Estadístico por Especie")
    resumen = df_f.groupby('Especie')[['Longitud Sépalo', 'Ancho Sépalo', 'Longitud Pétalo', 'Ancho Pétalo']].mean().round(2)
    st.dataframe(resumen, use_container_width=True)
    st.caption("Interpretación: Respaldo numérico exacto con los promedios en centímetros de cada especie.")
