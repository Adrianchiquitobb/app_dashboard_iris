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
    iris = load_iris(as_frame=True)
    df_iris = iris.frame
    df_iris.columns = ['Largo del Sépalo (cm)', 'Ancho del Sépalo (cm)', 'Largo del Pétalo (cm)', 'Ancho del Pétalo (cm)', 'target']
    df_iris['Tipo de Flor'] = df_iris['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    
    datos_geo = {
        'País': ['Canadá', 'Rusia', 'Japón', 'Estados Unidos', 'Reino Unido', 'Países Bajos', 'Alemania', 'Francia', 'México'],
        'ISO': ['CAN', 'RUS', 'JPN', 'USA', 'GBR', 'NLD', 'DEU', 'FRA', 'MEX'],
        'Tipo de Flor': ['Setosa', 'Setosa', 'Setosa', 'Versicolor', 'Versicolor', 'Virginica', 'Virginica', 'Virginica', 'Versicolor'],
        'Flores Registradas': [420, 380, 290, 510, 180, 340, 210, 195, 110]
    }
    df_geo = pd.DataFrame(datos_geo)
    return df_iris, df_geo

df_iris, df_geo = cargar_datos()

COLORES_FLOR = {'Setosa': '#2E7D32', 'Versicolor': '#FB8C00', 'Virginica': '#8E24AA'}

# =====================================================================
# 3. BARRA LATERAL (Filtros Sencillos)
# =====================================================================
st.sidebar.header("🔎 Explora los Datos")
st.sidebar.write("Filtra qué tipos de flor quieres analizar en todas las gráficas:")

especies_sel = st.sidebar.multiselect(
    "Tipos de flor Iris:",
    options=df_iris['Tipo de Flor'].unique(),
    default=df_iris['Tipo de Flor'].unique()
)

variable_bigote = st.sidebar.selectbox(
    "Parte de la flor a medir (KPI 4 - Bigote):",
    options=['Largo del Pétalo (cm)', 'Ancho del Pétalo (cm)', 'Largo del Sépalo (cm)', 'Ancho del Sépalo (cm)']
)

df_f = df_iris[df_iris['Tipo de Flor'].isin(especies_sel)]
df_geo_f = df_geo[df_geo['Tipo de Flor'].isin(especies_sel)]

if df_f.empty:
    st.warning("Por favor, selecciona al menos un tipo de flor en el menú de la izquierda.")
    st.stop()

# =====================================================================
# 4. TÍTULO LLAMATIVO Y PREGUNTA DE INVESTIGACIÓN CORTA
# =====================================================================
st.markdown(
    """
    <div style='background-color:#2E7D32; padding:25px; border-radius:12px; margin-bottom:25px;'>
        <h1 style='color:white; text-align:center; margin:0; font-size:36px;'>🌸 El Mundo de las Flores Iris: Tamaños, Formas y Presencia Global</h1>
        <p style='color:#E8F5E9; text-align:center; font-size:20px; margin-top:12px;'>
            <b>Pregunta de Investigación:</b> ¿En qué se diferencian las flores Iris y dónde habitan?
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================================
# 5. FILA 1: KPI 1 (Mapa Geográfico - Ancho Completo)
# =====================================================================
st.subheader("KPI 1: ¿En qué países se encuentran estas flores? (Mapa Geográfico)")
fig_mapa = px.choropleth(
    df_geo_f,
    locations="ISO",
    color="Flores Registradas",
    hover_name="País",
    custom_data=["Tipo de Flor", "Flores Registradas"],
    color_continuous_scale=px.colors.sequential.Greens
)
fig_mapa.update_traces(
    hovertemplate="<b>País:</b> %{hovertext}<br><b>Flor predominante:</b> %{customdata[0]}<br><b>Cantidad registrada:</b> %{customdata[1]} flores<extra></extra>"
)
fig_mapa.update_layout(margin={"r":0,"t":10,"l":0,"b":0}, height=420)
st.plotly_chart(fig_mapa, use_container_width=True)
st.caption("💡 Cómo leerlo: Entre más oscuro es el color verde del país, más flores Iris se han registrado ahí. Pasa el cursor sobre cualquier país pintado para ver el detalle.")

st.divider()

# =====================================================================
# 6. FILA 2: KPI 2 (Regresión Lineal) y KPI 3 (Mapa de Calor)
# =====================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("KPI 2: ¿Pétalo más largo = más ancho? (Regresión Lineal)")
    fig_reg = px.scatter(
        df_f,
        x="Largo del Pétalo (cm)",
        y="Ancho del Pétalo (cm)",
        color="Tipo de Flor",
        color_discrete_map=COLORES_FLOR,
        trendline="ols"
    )
    fig_reg.update_traces(
        selector=dict(mode='markers'),
        hovertemplate="<b>Largo del pétalo:</b> %{x} cm<br><b>Ancho del pétalo:</b> %{y} cm<extra></extra>"
    )
    fig_reg.update_traces(
        selector=dict(mode='lines'),
        hovertemplate="<b>Tendencia del pétalo:</b> %{y:.2f} cm de ancho a los %{x:.2f} cm de largo<extra></extra>"
    )
    fig_reg.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
    st.plotly_chart(fig_reg, use_container_width=True)
    st.caption("💡 Cómo leerlo: Cada punto es una flor real. Las líneas rectas suben, confirmando que entre más largo es el pétalo, también es más ancho.")

with col2:
    st.subheader("KPI 3: ¿Qué partes crecen juntas? (Mapa de Calor)")
    cols_medidas = ['Largo del Sépalo (cm)', 'Ancho del Sépalo (cm)', 'Largo del Pétalo (cm)', 'Ancho del Pétalo (cm)']
    corr = df_f[cols_medidas].corr()
    fig_heat = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    fig_heat.update_traces(
        hovertemplate="<b>Parte 1:</b> %{x}<br><b>Parte 2:</b> %{y}<br><b>Nivel de relación:</b> %{z:.2f} (cercano a 1 = crecen juntas)<extra></extra>"
    )
    fig_heat.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("💡 Cómo leerlo: Los cuadros verde oscuro (cercanos a 1) indican partes que crecen al mismo ritmo. Los tonos claros o rojizos indican que no dependen entre sí.")

st.divider()

# =====================================================================
# 7. FILA 3: KPI 4 (Gráfica de Bigote) y KPI 5 (Barras Comparativas)
# =====================================================================
col3, col4 = st.columns(2)

with col3:
    nombre_limpio = variable_bigote.replace(" (cm)", "")
    st.subheader(f"KPI 4: ¿Qué tan uniforme es el {nombre_limpio}? (Gráfica de Bigote)")
    fig_box = px.box(
        df_f,
        x="Tipo de Flor",
        y=variable_bigote,
        color="Tipo de Flor",
        color_discrete_map=COLORES_FLOR,
        points="all"
    )
    fig_box.update_traces(
        hoveron="points",
        hovertemplate=f"<b>Tipo de Flor:</b> %{{x}}<br><b>{nombre_limpio}:</b> %{{y}} cm<extra></extra>"
    )
    fig_box.update_layout(margin={"r":0,"t":20,"l":0,"b":0}, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption(f"💡 Cómo leerlo: Una caja pequeña significa que casi todas las flores tienen un {nombre_limpio.lower()} muy parecido. Pasa el cursor sobre cualquier punto para ver su medida exacta.")

with col4:
    st.subheader("KPI 5: Comparación de medidas promedio por flor")
    promedios = df_f.groupby('Tipo de Flor')[cols_medidas].mean().round(2).reset_index()
    promedios_largo = promedios.melt(id_vars='Tipo de Flor', var_name='Parte de la Flor', value_name='Centímetros')
    fig_barras = px.bar(
        promedios_largo,
        x="Parte de la Flor",
        y="Centímetros",
        color="Tipo de Flor",
        barmode="group",
        color_discrete_map=COLORES_FLOR
    )
    fig_barras.update_traces(
        hovertemplate="<b>Parte:</b> %{x}<br><b>Tamaño promedio:</b> %{y} cm<extra></extra>"
    )
    fig_barras.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
    st.plotly_chart(fig_barras, use_container_width=True)
    st.caption("💡 Cómo leerlo: Permite comparar de un vistazo qué tipo de flor es más grande en cada parte. El sépalo es la hoja externa que sostiene a la flor.")

st.divider()

# =====================================================================
# 8. FILA 4: KPI 6 (Gráfico de Dona) y KPI 7 (Ranking de Países)
# =====================================================================
col5, col6 = st.columns(2)

with col5:
    st.subheader("KPI 6: ¿Cuántas flores hay de cada tipo?")
    conteo = df_f['Tipo de Flor'].value_counts().reset_index()
    conteo.columns = ['Tipo de Flor', 'Cantidad']
    fig_dona = px.pie(
        conteo,
        values='Cantidad',
        names='Tipo de Flor',
        color='Tipo de Flor',
        color_discrete_map=COLORES_FLOR,
        hole=0.5
    )
    fig_dona.update_traces(
        hovertemplate="<b>Tipo de Flor:</b> %{label}<br><b>Cantidad:</b> %{value} flores<br><b>Porcentaje:</b> %{percent}<extra></extra>"
    )
    fig_dona.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
    st.plotly_chart(fig_dona, use_container_width=True)
    st.caption("💡 Cómo leerlo: Muestra si el estudio tiene la misma cantidad de flores de cada tipo para que la comparación sea justa.")

with col6:
    st.subheader("KPI 7: Top países con más flores registradas")
    fig_top = px.bar(
        df_geo_f,
        x="Flores Registradas",
        y="País",
        orientation="h",
        color="Tipo de Flor",
        color_discrete_map=COLORES_FLOR
    )
    fig_top.update_traces(
        hovertemplate="<b>País:</b> %{y}<br><b>Total registrado:</b> %{x} flores<extra></extra>"
    )
    fig_top.update_yaxes(categoryorder='total ascending')
    fig_top.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
    st.plotly_chart(fig_top, use_container_width=True)
    st.caption("💡 Cómo leerlo: Ordena de mayor a menor los países con más registros botánicos, mostrando qué especie domina en cada lugar.")

st.divider()

# =====================================================================
# 9. FILA 5: KPI 8 (Histograma) y KPI 9 (Récords de Tamaño)
# =====================================================================
col7, col8 = st.columns(2)

with col7:
    st.subheader("KPI 8: ¿Cuál es el largo de pétalo más común? (Histograma)")
    fig_hist = px.histogram(
        df_f,
        x="Largo del Pétalo (cm)",
        color="Tipo de Flor",
        color_discrete_map=COLORES_FLOR,
        barmode="group",
        nbins=15
    )
    fig_hist.update_traces(
        marker_line_width=1.5,
        marker_line_color="white",
        hovertemplate="<b>Rango de tamaño:</b> %{x} cm<br><b>Cantidad de flores:</b> %{y} flores<extra></extra>"
    )
    fig_hist.update_layout(yaxis_title="Cantidad de Flores", margin={"r":0,"t":20,"l":0,"b":0})
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("💡 Cómo leerlo: Este histograma agrupa las flores por rangos de tamaño. Las barras más altas indican las medidas que más se repiten sin encimarse entre especies.")

with col8:
    st.subheader("KPI 9: Récords de pétalo (Más pequeño, promedio y más grande)")
    records = df_f.groupby('Tipo de Flor')['Largo del Pétalo (cm)'].agg(['min', 'mean', 'max']).round(2).reset_index()
    records.columns = ['Tipo de Flor', 'Más pequeño (cm)', 'Promedio (cm)', 'Más grande (cm)']
    records_largo = records.melt(id_vars='Tipo de Flor', var_name='Medida', value_name='Centímetros')
    fig_rec = px.bar(
        records_largo,
        x="Tipo de Flor",
        y="Centímetros",
        color="Medida",
        barmode="group",
        color_discrete_sequence=["#A5D6A7", "#43A047", "#1B5E20"]
    )
    fig_rec.update_traces(
        hovertemplate="<b>Tipo de Flor:</b> %{x}<br><b>Medida:</b> %{y} cm<extra></extra>"
    )
    fig_rec.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
    st.plotly_chart(fig_rec, use_container_width=True)
    st.caption("💡 Cómo leerlo: Muestra el límite inferior y superior de cada especie para saber rápidamente hasta dónde puede llegar a crecer su pétalo.")
