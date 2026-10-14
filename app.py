import sys
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================================
# 1. DEFINICIÓN DE CLASES PERSONALIZADAS REQUERIDAS POR EL ARCHIVO .PKL
# =====================================================================
MAPA_TAMANO_MERCADO = {
    'pequeño': 1, 'pequeno': 1, 'pequeña': 1, 'pequena': 1, 'chico': 1, 'small': 1,
    'mediano': 2, 'mediana': 2, 'median': 2, 'medio': 2, 'medium': 2,
    'grande': 3, 'large': 3,
}

ETIQUETAS_TRAMO = np.array(['Baja', 'Media', 'Alta'])

def tramos_a_etiquetas(X):
    """Convierte los tramos 0/1/2 del discretizador en Baja/Media/Alta."""
    return ETIQUETAS_TRAMO[np.asarray(X).astype(int)]

class IngenieriaCaracteristicas(BaseEstimator, TransformerMixin):
    """Crea interacciones, polinomios y convierte códigos numéricos en categorías."""
    def __init__(self, grado=2, columnas_polinomio=('paginas', 'nomina'),
                 columnas_codigo=('idmercado', 'promo')):
        self.grado = grado
        self.columnas_polinomio = columnas_polinomio
        self.columnas_codigo = columnas_codigo

    def fit(self, X, y=None):
        self.columnas_salida_ = self.transform(X.iloc[:1]).columns.to_numpy()
        return self

    def get_feature_names_out(self, input_features=None):
        return self.columnas_salida_

    def transform(self, X):
        X = X.copy()
        # 1) Interacción: correos enviados x tamaño del mercado
        tamano = X['tamamer'].astype(str).str.strip().str.lower()
        nivel = tamano.map(MAPA_TAMANO_MERCADO)
        if nivel.isna().any():
            raise ValueError(
                f"Valores de 'tamamer' sin nivel asignado: {sorted(tamano[nivel.isna()].unique())}. "
                "Agréguelos a MAPA_TAMANO_MERCADO."
            )
        X['correo_x_tamamer'] = X['correo'] * nivel

        # 2) Polinomios para capturar rendimientos decrecientes
        for col in self.columnas_polinomio:
            for g in range(2, self.grado + 1):
                X[f"{col}^{g}"] = X[col].astype(float) ** g

        # 3) Los códigos se tratan como categorías, no como cantidades
        for col in self.columnas_codigo:
            X[col] = X[col].astype(str)

        return X

# Inyectamos la clase y función en __main__ porque el .pkl fue creado en una celda de Colab
setattr(sys.modules['__main__'], 'IngenieriaCaracteristicas', IngenieriaCaracteristicas)
setattr(sys.modules['__main__'], 'tramos_a_etiquetas', tramos_a_etiquetas)
setattr(sys.modules['__main__'], 'ETIQUETAS_TRAMO', ETIQUETAS_TRAMO)
setattr(sys.modules['__main__'], 'MAPA_TAMANO_MERCADO', MAPA_TAMANO_MERCADO)

# =====================================================================
# 2. CONFIGURACIÓN DE LA INTERFAZ STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="Pronóstico de Ventas - Catálogos",
    page_icon="🛍️",
    layout="wide"
)

PROMEDIO_HISTORICO = 50978.38
RMSE_PRUEBA = 10286.66
COLUMNAS_REQUERIDAS = [
    'edadloc', 'correo', 'paginas', 'telefono',
    'impresa', 'servicio', 'nomina', 'idmercado', 'tamamer', 'promo'
]

# =====================================================================
# 3. CARGA AUTOMÁTICA DEL ARCHIVO .PKL
# =====================================================================
@st.cache_resource
def cargar_modelo_pkl():
    archivos_pkl = glob.glob("*.pkl")
    if "modelo_catalogo.pkl" in archivos_pkl:
        ruta = "modelo_catalogo.pkl"
    elif len(archivos_pkl) > 0:
        ruta = archivos_pkl[0]
    else:
        raise FileNotFoundError("No se encontró ningún archivo .pkl en el repositorio.")
    return joblib.load(ruta), ruta

try:
    modelo, nombre_archivo_pkl = cargar_modelo_pkl()
    modelo_listo = True
except Exception as err:
    modelo_listo = False
    detalle_error = str(err)

# =====================================================================
# 4. ENCABEZADO Y FICHA TÉCNICA LATERAL
# =====================================================================
st.title("🛍️ Sistema Empresarial de Pronóstico de Ventas (`ropamujer`)")
st.markdown(
    "Aplicación en producción bajo la metodología **CRISP-DM**. "
    "Ejecuta el Pipeline completo (`IngenieriaCaracteristicas` → `ColumnTransformer` → `GradientBoostingRegressor`) "
    "entrenado con las **1,600 sucursales históricas**."
)

with st.sidebar:
    st.header("📊 Ficha Técnica (CRISP-DM)")
    if modelo_listo:
        st.success(f"Modelo activo: `{nombre_archivo_pkl}`")
    st.markdown("""
    - **Algoritmo:** Gradient Boosting Regressor
    - **Árboles (`n_estimators`):** 200
    - **Profundidad (`max_depth`):** 2
    - **Tasa aprendizaje (`lr`):** 0.03
    - **Submuestreo (`subsample`):** 0.8
    """)
    st.divider()
    st.markdown("""
    **Métricas de Validación:**
    - **Línea Base (Fase 1):** $17,552.77 USD
    - **RMSE CV (Fase 4):** $10,902.96 USD
    - **RMSE Prueba (Fase 5):** $10,286.66 USD
    """)
    st.divider()
    st.subheader("🔑 Impacto por Permutación")
    st.caption("Aumento del error si se omite la variable:")
    st.markdown("""
    1. `servicio`: **+$3,816.31 USD**
    2. `impresa`: **+$2,232.00 USD**
    3. `correo`: **+$2,032.78 USD**
    4. `edadloc`: **+$1,710.76 USD**
    5. `paginas`: **+$644.08 USD**
    """)

if not modelo_listo:
    st.error(f"❌ Error al cargar el archivo `.pkl`: {detalle_error}")
    st.stop()

# =====================================================================
# 5. PESTAÑAS DE OPERACIÓN
# =====================================================================
tab1, tab2 = st.tabs([
    "🎯 Simulador de Sucursal Individual",
    "📂 Pronóstico por Lote (400 Nuevas Zonas)"
])

# ---------------------------------------------------------------------
# PESTAÑA 1: SIMULACIÓN INDIVIDUAL
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Simulación de Recursos Comerciales por Sucursal")
    st.markdown("Ingresa las características operativas y de marketing de la sucursal:")

    with st.form("formulario_individual"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("##### 📢 Marketing y Catálogo")
            correo = st.number_input("Catálogos enviados por correo (`correo`)", min_value=500, max_value=30000, value=10042, step=250)
            paginas = st.number_input("Número de páginas del catálogo (`paginas`)", min_value=10, max_value=250, value=68, step=2)
            impresa = st.number_input("Inversión en publicidad impresa USD (`impresa`)", min_value=1000.0, max_value=100000.0, value=22609.49, step=500.0)
            promo = st.selectbox("Código de promoción (`promo`)", options=[1, 2, 3], index=1)

        with c2:
            st.markdown("##### 🏪 Capacidad y Antigüedad")
            servicio = st.number_input("Representantes de servicio (`servicio`)", min_value=1, max_value=150, value=30, step=1)
            telefono = st.number_input("Operadores telefónicos (`telefono`)", min_value=1, max_value=120, value=35, step=1)
            edadloc = st.number_input("Antigüedad del local en años (`edadloc`)", min_value=1, max_value=60, value=11, step=1)

        with c3:
            st.markdown("##### 🌍 Mercado y Nómina")
            tamamer = st.selectbox("Tamaño del mercado (`tamamer`)", options=["Grande", "Median", "Pequeño"], index=0)
            idmercado = st.selectbox("ID de mercado (`idmercado`)", options=[1, 2, 3, 4, 5, 6, 7, 8, 9], index=7)
            nomina = st.number_input("Nómina anual USD (`nomina`)", min_value=1000, max_value=200000, value=66307, step=1000)

        calcular = st.form_submit_button("🚀 Estimar Ventas Anuales", use_container_width=True)

    if calcular:
        datos_sucursal = pd.DataFrame([{
            'edadloc': int(edadloc),
            'correo': int(correo),
            'paginas': int(paginas),
            'telefono': int(telefono),
            'impresa': float(impresa),
            'servicio': int(servicio),
            'nomina': int(nomina),
            'idmercado': int(idmercado),
            'tamamer': str(tamamer),
            'promo': int(promo)
        }])

        venta_estimada = float(modelo.predict(datos_sucursal)[0])
        dif_promedio = venta_estimada - PROMEDIO_HISTORICO
        pct_promedio = (dif_promedio / PROMEDIO_HISTORICO) * 100

        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.metric(
            label="💰 Ventas Anuales Proyectadas (`ropamujer`)",
            value=f"${venta_estimada:,.2f} USD",
            delta=f"{dif_promedio:+,.2f} USD ({pct_promedio:+.1f}%) vs. Línea Base"
        )
        k2.metric(
            label="📉 Escenario Conservador (-1 RMSE)",
            value=f"${max(0.0, venta_estimada - RMSE_PRUEBA):,.2f} USD"
        )
        k3.metric(
            label="📈 Escenario Optimista (+1 RMSE)",
            value=f"${venta_estimada + RMSE_PRUEBA:,.2f} USD"
        )

# ---------------------------------------------------------------------
# PESTAÑA 2: PRONÓSTICO CIEGO POR LOTE (FASE 6)
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Pronóstico Ciego para Nuevas Sucursales (`test.csv` / `.xlsx`)")
    st.markdown(
        "Carga el archivo con las **400 nuevas zonas** para aplicar el modelo consolidado "
        "y descargar el entregable oficial con las columnas `idloc` y `ropamujer`."
    )

    archivo = st.file_uploader("Selecciona tu archivo (.csv o .xlsx)", type=["csv", "xlsx"])

    if archivo is not None:
        try:
            if archivo.name.endswith(".csv"):
                df_lote = pd.read_csv(archivo)
            else:
                df_lote = pd.read_excel(archivo)

            st.success(f"✅ Archivo cargado: **{df_lote.shape[0]} registros** encontrados.")

            cols_faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df_lote.columns]
            if cols_faltantes:
                st.error(f"❌ El archivo no contiene las siguientes columnas obligatorias: `{cols_faltantes}`")
            else:
                ids = df_lote['idloc'] if 'idloc' in df_lote.columns else np.arange(1, len(df_lote) + 1)
                X_nuevos = df_lote[COLUMNAS_REQUERIDAS]

                predicciones = modelo.predict(X_nuevos)

                entregable = pd.DataFrame({
                    'idloc': ids,
                    'ropamujer': predicciones
                })

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Zonas Pronosticadas", f"{len(entregable):,}")
                m2.metric("Promedio Proyectado", f"${entregable['ropamujer'].mean():,.2f}")
                m3.metric("Mínimo Proyectado", f"${entregable['ropamujer'].min():,.2f}")
                m4.metric("Máximo Proyectado", f"${entregable['ropamujer'].max():,.2f}")

                st.markdown("#### Vista Previa del Entregable Oficial")
                st.dataframe(entregable.head(15), use_container_width=True)

                csv_descarga = entregable.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar predicciones_catalogo_400zonas.csv",
                    data=csv_descarga,
                    file_name="predicciones_catalogo_400zonas.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
