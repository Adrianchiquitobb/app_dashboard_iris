import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

# =====================================================================
# 1. CONFIGURACIÓN DE PÁGINA (Pantalla Completa)
# =====================================================================
st.set_page_config(
    page_title="El Mundo de las Flores Iris",
    page_icon="🌸",
    layout="wide"
)

# =====================================================================
# 2. CARGA DE DATASETS
# =====================================================================
@st.cache_data
def cargar_datos():
    # Dataset 1: Medidas reales en centímetros
    iris = load_iris(as_frame=True)
    df_iris = iris.frame
    df_iris.columns = ['Largo del Sépalo (cm)', 'Ancho del Sépalo (cm)', 'Largo del Pétalo (cm)', 'Ancho del Pétalo (cm)', 'target']
    df_iris['Tipo de Flor'] = df_iris['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    
    # Dataset 2: Presencia en el mundo
    datos_geo = {
        'País': ['Canada', 'Russia', 'Japan', 'United States', 'United Kingdom', 'Netherlands', 'Germany', 'France', 'Mexico'],
        'ISO': ['CAN', 'RUS', 'JPN', 'USA', 'GBR', 'NLD', 'DEU', 'FRA', 'MEX'],
        'Tipo de Flor': ['Setosa', 'Setosa', 'Setosa', 'Versicolor', 'Versicolor', 'Virginica', 'Virginica', 'Virginica', 'Versicolor'],
        'Flores Registradas': [420, 380, 290, 510, 180, 340, 210, 195, 110]
    }
    df_geo = pd.DataFrame(datos_geo)
    return df_iris, df_geo

df_iris, df_geo = cargar_datos()

# =====================================================================
# 3. BARRA LATERAL (Filtros Sencillos)
# =====================================================================
st.sidebar.header("🔎 Explora los Datos")
st.sidebar.write("Usa estos controles para filtrar qué flores quieres ver en las gráficas:")

especies_sel = st.sidebar.multiselect(
    "Elige los tipos de flor Iris:",
    options=df_iris['Tipo de Flor'].unique(),
    default=df_iris['Tipo de Flor'].unique()
)

variable_bigote = st.sidebar.selectbox(
    "Parte de la flor a comparar (Gráfica de Bigote):",
    options=['Largo del Pétalo (cm)', 'Ancho del Pétalo (cm)', 'Largo del Sépalo (cm)', 'Ancho del Sépalo (cm)']
)

df_f = df_iris[df_iris['Tipo de Flor'].isin(especies_sel)]
df_geo_f = df_geo[df_geo['Tipo de Flor'].isin(especies_sel)]

if df_f.empty:
    st.warning("Por favor, selecciona al menos un tipo de flor en el menú de la izquierda.")
    st.stop()

# =====================================================================
# 4. TÍTULO LLAMATIVO Y PREGUNTA DE INVESTIGACIÓN
# =====================================================================
st.markdown(
    """
    <div style='background-color:#2E7D32; padding:25px; border-radius:12px; margin-bottom:25px;'>
        <h1 style='color:white; text-align:center; margin:0; font-size:36px;'>🌸 El Mundo de las Flores Iris: Tamaños, Formas y Presencia Global</h1>
        <p style='color:#E8F5E9; text-align:center; font-size:19px; margin-top:12px;'>
            <b>Pregunta de Investigación:</b> ¿Cómo podemos distinguir fácilmente los 3 tipos de flores Iris por su tamaño y en qué países es más común encontrarlas?
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================================
# 5. FILA 1: KPIs 1, 2 y 3 (Tarjetas de Resumen Fácil)
# =====================================================================
k1, k2, k3 = st.columns(3)
with k1:
    st.metric(label="KPI 1: Total de Flores Estudiadas", value=f"{len(df_f)} flores")
    st.caption("Cantidad total de flores que estás viendo en este momento.")
with k2:
    st.metric(label="KPI 2: Largo Promedio del Pétalo", value=f"{df_f['Largo del Pétalo (cm)'].mean():.2f} cm")
    st.caption("El pétalo (parte colorida) es la mejor pista para diferenciarlas.")
with k3:
    st.metric(label="KPI 3: Largo Promedio del Sépalo", value=f"{df_f['Largo del Sépalo (cm)'].mean():.2f} cm")
    st.caption("El sépalo es la hoja externa que protege y sostiene a la flor.")

st.divider()

# =====================================================================
# 6. FILA 2: KPI 4 (Mapa Geográfico) y KPI 5 (Regresión Lineal)
# =====================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("KPI 4: ¿Dónde viven estas flores? (Mapa Geográfico)")
    fig_mapa = px.choropleth(
        df_geo_f,
        locations="ISO",
        color="Flores Registradas",
        hover_name="País",
        hover_data=["Tipo de Flor"],
        color_continuous_scale=px.colors.sequential.Greens
    )
    fig_mapa.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
    st.plotly_chart(fig_mapa, use_container_width=True)
    st.caption("💡 Cómo leerlo: Entre más oscuro es el verde de un país, más flores de este tipo se han encontrado ahí. Pasa el cursor sobre un país para ver el detalle.")

with col2:
    st.subheader("KPI 5: ¿Pétalo más largo = más ancho? (Regresión Lineal)")
    fig_reg = px.scatter(
        df_f,
        x="Largo del Pétalo (cm)",
        y="Ancho del Pétalo (cm)",
        color="Tipo de Flor",
        trendline="ols"
    )
    fig_reg.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
    st.plotly_chart(fig_reg, use_container_width=True)
    st.caption("💡 Cómo leerlo: Las líneas rectas suben de izquierda a derecha, confirmando que entre más largo es el pétalo, también es más ancho. Nota cómo la Setosa es la más pequeña.")

# =====================================================================
# 7. FILA 3: KPI 6 (Mapa de Calor) y KPI 7 (Gráfica de Bigote)
# =====================================================================
col3, col4 = st.columns(2)

with col3:
    st.subheader("KPI 6: ¿Qué partes crecen juntas? (Mapa de Calor)")
    corr = df_f[['Largo del Sépalo (cm)', 'Ancho del Sépalo (cm)', 'Largo del Pétalo (cm)', 'Ancho del Pétalo (cm)']].corr()
    fig_heat = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    fig_heat.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("💡 Cómo leerlo: Los cuadros verde oscuro (cercanos a 1) muestran partes que crecen al mismo tiempo. Los tonos claros o rojizos indican que no dependen entre sí.")

with col4:
    st.subheader(f"KPI 7: ¿Qué tan parejos son sus tamaños? (Gráfica de Bigote)")
    fig_box = px.box(
        df_f,
        x="Tipo de Flor",
        y=variable_bigote,
        color="Tipo de Flor",
        points="all"
    )
    fig_box.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("💡 Cómo leerlo: Una caja pequeña significa que casi todas las flores de ese tipo miden lo mismo. Los puntos sueltos fuera de las líneas son flores inusualmente grandes o pequeñas.")

st.divider()

# =====================================================================
# 8. FILA 4: KPI 8 (Gráfico de Dona) y KPI 9 (Tabla Comparativa)
# =====================================================================
col5, col6 = st.columns(2)

with col5:
    st.subheader("KPI 8: ¿Cuántas flores hay de cada tipo?")
    conteo = df_f['Tipo de Flor'].value_counts().reset_index()
    conteo.columns = ['Tipo de Flor', 'Cantidad']
    fig_dona = px.pie(
        conteo,
        values='Cantidad',
        names='Tipo de Flor',
        hole=0.5
    )
    fig_dona.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
    st.plotly_chart(fig_dona, use_container_width=True)
    st.caption("💡 Cómo leerlo: Muestra que el estudio está perfectamente equilibrado, con la misma cantidad de flores analizadas por cada especie.")

with col6:
    st.subheader("KPI 9: Tabla Comparativa de Tamaños Promedio")
    resumen = df_f.groupby('Tipo de Flor')[['Largo del Sépalo (cm)', 'Ancho del Sépalo (cm)', 'Largo del Pétalo (cm)', 'Ancho del Pétalo (cm)']].mean().round(2)
    st.dataframe(resumen, use_container_width=True)
    st.caption("💡 Cómo leerlo: Resumen exacto en centímetros para comparar de un vistazo qué tipo de flor es más grande (Virginica) y cuál es más pequeña (Setosa).")
