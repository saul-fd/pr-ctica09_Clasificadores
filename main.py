"""
Dashboard Práctica 09 - Clasificadores de Rotación de Personal
Equipo 12: Saul Fabila, Sergio Sánchez, Diana Vélez
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── CONFIG PÁGINA ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Práctica 09 · Clasificadores",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS PERSONALIZADO ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fondo oscuro */
.stApp {
    background-color: #0b0f1a;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 14px;
}

/* Títulos */
h1 { font-family: 'Space Mono', monospace; color: #f8fafc !important; letter-spacing: -1px; }
h2 { font-family: 'Space Mono', monospace; color: #7dd3fc !important; font-size: 1.3rem !important; }
h3 { color: #94a3b8 !important; font-weight: 500; font-size: 1rem !important; }

/* Métricas */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="metric-container"] label { color: #64748b !important; font-size: 12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #7dd3fc !important;
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 12px !important; }

/* Cards */
.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}
.card-accent {
    border-left: 3px solid #7dd3fc;
}
.card-warning {
    border-left: 3px solid #fbbf24;
}
.card-success {
    border-left: 3px solid #34d399;
}
.card-danger {
    border-left: 3px solid #f87171;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
    margin-right: 4px;
}
.badge-blue  { background: #1e3a5f; color: #7dd3fc; }
.badge-green { background: #14432a; color: #34d399; }
.badge-yellow{ background: #3d2e00; color: #fbbf24; }
.badge-red   { background: #3d1010; color: #f87171; }

/* Header hero */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(125,211,252,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 8px 0;
    letter-spacing: -1px;
}
.hero-sub {
    color: #64748b;
    font-size: 14px;
    margin: 0;
}
.hero-tag {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #7dd3fc;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Step cards */
.step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 8px;
}
.step-num {
    background: #1e3a5f;
    color: #7dd3fc;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    border-radius: 6px;
    padding: 4px 8px;
    min-width: 32px;
    text-align: center;
}

/* Tabs override */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #7dd3fc !important;
}

/* Expander */
.streamlit-expanderHeader { color: #94a3b8 !important; }

/* Divider */
hr { border-color: #1e293b !important; }

/* Selectbox */
.stSelectbox select, .stSelectbox [data-baseweb="select"] {
    background-color: #111827;
    color: #e2e8f0;
    border-color: #1e293b;
}

/* Table */
.dataframe { background: #111827 !important; color: #e2e8f0 !important; }
.dataframe th { background: #1e293b !important; color: #7dd3fc !important; }
.dataframe td { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ─── DATOS HARDCODEADOS DEL NOTEBOOK ──────────────────────────────────────────

RESULTADOS = {
    "SVM": {
        "accuracy": 0.8605, "precision": 0.5556, "recall": 0.2174, "f1": 0.3125,
        "tp": 10, "fp": 8, "fn": 36, "tn": 242,
        "kernel": "linear", "desc": "Support Vector Machine"
    },
    "KNN": {
        "accuracy": 0.8299, "precision": 0.3684, "recall": 0.1522, "f1": 0.2157,
        "tp": 7, "fp": 12, "fn": 39, "tn": 238,
        "kernel": "k=5", "desc": "K-Nearest Neighbors"
    },
    "Regresión Logística": {
        "accuracy": 0.8707, "precision": 0.6250, "recall": 0.2174, "f1": 0.3226,
        "tp": 10, "fp": 6, "fn": 36, "tn": 244,
        "kernel": "max_iter=1000", "desc": "Logistic Regression"
    },
    "Árbol de Decisión": {
        "accuracy": 0.8435, "precision": 0.4138, "recall": 0.2609, "f1": 0.3200,
        "tp": 12, "fp": 17, "fn": 34, "tn": 233,
        "kernel": "max_depth=5", "desc": "Decision Tree"
    },
    "Random Forest": {
        "accuracy": 0.8741, "precision": 0.7143, "recall": 0.2174, "f1": 0.3333,
        "tp": 10, "fp": 4, "fn": 36, "tn": 246,
        "kernel": "n=100, depth=10", "desc": "Random Forest"
    },
    "Naive Bayes": {
        "accuracy": 0.7891, "precision": 0.3415, "recall": 0.6087, "f1": 0.4375,
        "tp": 28, "fp": 54, "fn": 18, "tn": 196,
        "kernel": "Gaussian", "desc": "Gaussian Naive Bayes"
    },
    "Gamma (PYDRA)": {
        "accuracy": 0.8435, "precision": 0.4286, "recall": 0.1957, "f1": 0.2687,
        "tp": 9, "fp": 12, "fn": 37, "tn": 238,
        "kernel": "variante=7", "desc": "Gamma PYDRA (similitud)"
    },
    "Perceptrón": {
        "accuracy": 0.7925, "precision": 0.2593, "recall": 0.3043, "f1": 0.2800,
        "tp": 14, "fp": 40, "fn": 32, "tn": 210,
        "kernel": "l2, α=0.01", "desc": "Perceptron"
    },
}

# Dataset info
DATASET_INFO = {
    "total_registros": 1470,
    "n_features_originales": 35,
    "n_features_final": 46,
    "train_size": 1176,
    "test_size": 294,
    "clases": {"No renuncia": 1233, "Renuncia": 237},
    "nulos_generados": {
        "TrainingTimesLastYear": "5%",
        "YearsSinceLastPromotion": "15%",
        "YearsWithCurrManager": "12%",
    },
    "columnas_eliminadas": ["Over18", "EmployeeCount", "StandardHours", "EmployeeNumber"],
    "top5_riesgo": ["OverTime", "MaritalStatus_Single", "JobRole_Sales", "BusinessTravel", "DistanceFromHome"],
    "top5_retencion": ["JobLevel", "TotalWorkingYears", "YearsAtCompany", "MonthlyIncome", "Age"],
}

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Práctica 09")
    st.markdown("**Clasificadores de Attrition**")
    st.markdown("---")
    seccion = st.radio(
        "Navegar a:",
        ["🏠 Inicio", "📊 Dataset & Preprocesamiento", "🧪 Experimentación",
         "📈 Comparativa de Modelos", "🔬 Análisis Crítico"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#475569;'>
    <b>Equipo 12</b><br>
    Saul Fabila Domínguez<br>
    Sergio Iván Sánchez Portilla<br>
    Diana María Vélez Gallardo
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; color:#334155; margin-top:16px;'>
    Dataset: IBM HR Analytics<br>
    Objetivo: Predecir Attrition
    </div>
    """, unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────
PALETTE = {
    "blue": "#7dd3fc", "green": "#34d399", "yellow": "#fbbf24",
    "red": "#f87171", "purple": "#a78bfa", "orange": "#fb923c",
    "teal": "#2dd4bf", "pink": "#f472b6",
}
MODEL_COLORS = {
    "SVM": "#7dd3fc", "KNN": "#a78bfa", "Regresión Logística": "#34d399",
    "Árbol de Decisión": "#fbbf24", "Random Forest": "#f87171",
    "Naive Bayes": "#fb923c", "Gamma (PYDRA)": "#2dd4bf", "Perceptrón": "#f472b6",
}
BG = "#0b0f1a"
CARD_BG = "#111827"
GRID_COLOR = "#1e293b"

def plotly_theme(fig):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=CARD_BG,
        font=dict(family="DM Sans", color="#94a3b8", size=12),
        title_font=dict(family="Space Mono", color="#f8fafc", size=14),
        legend=dict(bgcolor="#111827", bordercolor=GRID_COLOR, borderwidth=1,
                    font=dict(color="#94a3b8")),
        margin=dict(l=16, r=16, t=48, b=16),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color="#64748b")),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color="#64748b")),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# INICIO
# ═══════════════════════════════════════════════════════════════════════════════
if seccion == "🏠 Inicio":
    st.markdown("""
    <div class='hero'>
        <p class='hero-tag'>Práctica 09 · Equipo 12</p>
        <h1 class='hero-title'>Clasificadores de<br>Rotación de Personal</h1>
        <p class='hero-sub'>IBM HR Analytics · 8 modelos evaluados · Predicción de Attrition</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", "1,470", "IBM HR Dataset")
    c2.metric("Modelos evaluados", "8", "clasificadores")
    c3.metric("Mejor accuracy", "87.41%", "Random Forest")
    c4.metric("Mejor F1-Score", "43.75%", "Naive Bayes")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("## 🎯 Objetivo")
        st.markdown("""
        <div class='card card-accent'>
            Predecir si un empleado va a <b style='color:#f87171'>renunciar</b> o no 
            (<code style='color:#7dd3fc'>Attrition = Yes/No</code>) utilizando datos del área de 
            Recursos Humanos de IBM.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 📋 Flujo de trabajo")
        pasos = [
            ("01", "Carga y exploración del dataset"),
            ("02", "Generación artificial de valores nulos"),
            ("03", "Imputación con media (SimpleImputer)"),
            ("04", "Codificación de variables categóricas"),
            ("05", "Escalado con StandardScaler"),
            ("06", "Split 80/20 estratificado"),
            ("07", "Entrenamiento de 8 clasificadores"),
            ("08", "Evaluación con Accuracy, Precision, Recall, F1"),
        ]
        for num, texto in pasos:
            st.markdown(f"""
            <div class='step'>
                <span class='step-num'>{num}</span>
                <span style='color:#cbd5e1; font-size:13px;'>{texto}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("## 🤖 Modelos evaluados")
        for nombre, datos in RESULTADOS.items():
            color = MODEL_COLORS[nombre]
            st.markdown(f"""
            <div class='card' style='padding:12px 16px; margin-bottom:6px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <span style='color:{color}; font-weight:600; font-size:13px;'>■ {nombre}</span>
                        <span style='color:#475569; font-size:11px; margin-left:8px;'>({datos["kernel"]})</span>
                    </div>
                    <span style='font-family:Space Mono; color:#94a3b8; font-size:12px;'>
                        Acc: {datos["accuracy"]:.4f}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Distribución de clases
        st.markdown("## ⚖️ Desbalance de clases")
        fig = go.Figure(go.Bar(
            x=["No renuncia (0)", "Renuncia (1)"],
            y=[1233, 237],
            marker_color=["#34d399", "#f87171"],
            text=["1,233\n(83.9%)", "237\n(16.1%)"],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=12, family="Space Mono"),
        ))
        fig.update_layout(
            title="Distribución de clases · Attrition",
            height=250, showlegend=False,
            paper_bgcolor=BG, plot_bgcolor=CARD_BG,
            font=dict(family="DM Sans", color="#94a3b8"),
            margin=dict(l=8, r=8, t=40, b=8),
            yaxis=dict(gridcolor=GRID_COLOR),
            xaxis=dict(linecolor=GRID_COLOR),
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET & PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════
elif seccion == "📊 Dataset & Preprocesamiento":
    st.markdown("## 📊 Dataset & Preprocesamiento")

    tab1, tab2, tab3, tab4 = st.tabs(["📁 Dataset", "🔧 Limpieza", "🔢 Codificación", "📐 Correlaciones"])

    # ── TAB 1: Dataset
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total registros", "1,470")
        col2.metric("Features originales", "35")
        col3.metric("Features finales", "46", "+11 (one-hot)")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Variables numéricas clave")
            vars_num = pd.DataFrame({
                "Variable": ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears",
                             "DistanceFromHome", "JobSatisfaction"],
                "Tipo": ["int", "int", "int", "int", "int", "ordinal"],
                "Rango": ["18-60", "1009-19999", "0-40", "0-40", "1-29", "1-4"],
            })
            st.dataframe(vars_num, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("#### Variables categóricas")
            vars_cat = pd.DataFrame({
                "Variable": ["Department", "JobRole", "MaritalStatus",
                             "BusinessTravel", "EducationField", "OverTime"],
                "Codificación": ["One-hot", "One-hot", "One-hot",
                                 "Ordinal (0-2)", "One-hot", "Binaria (0/1)"],
            })
            st.dataframe(vars_cat, use_container_width=True, hide_index=True)

        st.markdown("---")
        # Distribución target
        fig = go.Figure()
        labels = ["No renuncia", "Renuncia"]
        values = [1233, 237]
        colors = ["#34d399", "#f87171"]
        fig.add_trace(go.Pie(
            labels=labels, values=values,
            marker_colors=colors,
            hole=0.5,
            textinfo="label+percent",
            textfont=dict(color="#f8fafc", size=13),
        ))
        fig.update_layout(
            title="Variable objetivo: Attrition",
            height=320, paper_bgcolor=BG,
            font=dict(family="DM Sans", color="#94a3b8"),
            legend=dict(bgcolor=CARD_BG, bordercolor=GRID_COLOR),
            margin=dict(l=8, r=8, t=48, b=8),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: Limpieza
    with tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Valores nulos generados artificialmente")
            st.markdown("""
            <div class='card card-warning'>
            ⚠️ El dataset <b>no tenía valores perdidos originalmente</b>, por lo que se generaron 
            aleatoriamente para practicar técnicas de imputación.
            </div>
            """, unsafe_allow_html=True)

            nulos_data = pd.DataFrame({
                "Columna": ["TrainingTimesLastYear", "YearsSinceLastPromotion", "YearsWithCurrManager"],
                "Fracción nula": ["5%", "15%", "12%"],
                "N aprox. nulos": ["~74", "~221", "~176"],
                "Estrategia": ["Media", "Media", "Media"],
            })
            st.dataframe(nulos_data, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### Columnas eliminadas")
            st.markdown("""
            <div class='card card-danger'>
            🗑️ Columnas sin valor predictivo o con varianza cero:
            </div>
            """, unsafe_allow_html=True)
            for col in DATASET_INFO["columnas_eliminadas"]:
                motivo = {
                    "Over18": "Varianza cero (todos = 'Y')",
                    "EmployeeCount": "Constante = 1",
                    "StandardHours": "Constante = 80",
                    "EmployeeNumber": "Identificador único",
                }[col]
                st.markdown(f"""
                <div class='card' style='padding:10px 14px; margin-bottom:4px;'>
                    <code style='color:#f87171;'>{col}</code>
                    <span style='color:#64748b; font-size:12px; margin-left:8px;'>→ {motivo}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ⚠️ Nota importante: Data Leakage")
        st.markdown("""
        <div class='card card-warning'>
        El notebook reconoce explícitamente que la imputación completa se realizó <b>antes del split</b>, 
        lo cual introduce <b>data leakage</b>. El flujo correcto sería:<br><br>
        <code style='color:#7dd3fc;'>1. Split → 2. fit(X_train) → 3. transform(X_train) → 4. transform(X_test)</code><br><br>
        Esto aplica tanto al <b>SimpleImputer</b> como al <b>StandardScaler</b>.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 3: Codificación y Escalado
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Codificación binaria")
            bin_data = pd.DataFrame({
                "Variable": ["Attrition", "Gender", "OverTime"],
                "Mapeo": ["Yes→1, No→0", "Male→1, Female→0", "Yes→1, No→0"],
            })
            st.dataframe(bin_data, use_container_width=True, hide_index=True)

            st.markdown("#### Codificación ordinal")
            ord_data = pd.DataFrame({
                "Variable": ["BusinessTravel"],
                "Mapeo": ["Non-Travel→0, Travel_Rarely→1, Travel_Frequently→2"],
            })
            st.dataframe(ord_data, use_container_width=True, hide_index=True)

            st.markdown("#### One-Hot Encoding")
            ohe_data = pd.DataFrame({
                "Variable": ["Department", "EducationField", "JobRole", "MaritalStatus"],
                "Categorías": ["3", "6", "9", "3"],
            })
            st.dataframe(ohe_data, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### StandardScaler")
            st.markdown("""
            <div class='card card-accent'>
            Se aplicó <b>StandardScaler</b> a todas las variables numéricas para centrar 
            en μ≈0 y σ≈1, necesario para modelos sensibles a escala como SVM, KNN y Perceptrón.
            </div>
            """, unsafe_allow_html=True)

            # Mini visualización del efecto del escalado
            np.random.seed(42)
            original = np.random.normal(loc=50, scale=15, size=200)
            scaled = (original - original.mean()) / original.std()

            fig = make_subplots(rows=1, cols=2, subplot_titles=["Antes del escalado", "Después del escalado"])
            fig.add_trace(go.Histogram(x=original, nbinsx=20, marker_color="#7dd3fc",
                                       name="Original", opacity=0.8), row=1, col=1)
            fig.add_trace(go.Histogram(x=scaled, nbinsx=20, marker_color="#34d399",
                                       name="Escalado", opacity=0.8), row=1, col=2)
            fig.update_layout(
                height=250, showlegend=False, title="Efecto del StandardScaler (ejemplo)",
                paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                font=dict(color="#94a3b8", family="DM Sans"),
                margin=dict(l=8, r=8, t=48, b=8),
            )
            fig.update_xaxes(gridcolor=GRID_COLOR)
            fig.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div style='display:flex; gap:8px; margin-top:8px;'>
                <div class='card' style='flex:1; text-align:center;'>
                    <div style='font-family:Space Mono; color:#7dd3fc; font-size:1.2rem;'>80/20</div>
                    <div style='color:#64748b; font-size:11px;'>Train/Test split</div>
                </div>
                <div class='card' style='flex:1; text-align:center;'>
                    <div style='font-family:Space Mono; color:#34d399; font-size:1.2rem;'>1,176</div>
                    <div style='color:#64748b; font-size:11px;'>Train samples</div>
                </div>
                <div class='card' style='flex:1; text-align:center;'>
                    <div style='font-family:Space Mono; color:#fbbf24; font-size:1.2rem;'>294</div>
                    <div style='color:#64748b; font-size:11px;'>Test samples</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 4: Correlaciones
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔴 Top 5 factores que impulsan la RENUNCIA")
            factores_riesgo = [
                ("OverTime", 0.246),
                ("MaritalStatus_Single", 0.175),
                ("JobRole_Sales Representative", 0.148),
                ("BusinessTravel", 0.131),
                ("DistanceFromHome", 0.097),
            ]
            fig = go.Figure(go.Bar(
                x=[v for _, v in factores_riesgo],
                y=[n for n, _ in factores_riesgo],
                orientation="h",
                marker_color="#f87171",
                text=[f"{v:.3f}" for _, v in factores_riesgo],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=11),
            ))
            fig.update_layout(
                height=250, paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                font=dict(color="#94a3b8"), margin=dict(l=8, r=60, t=8, b=8),
                xaxis=dict(gridcolor=GRID_COLOR), yaxis=dict(gridcolor=GRID_COLOR),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 🟢 Top 5 factores que RETIENEN al empleado")
            factores_retencion = [
                ("JobLevel", -0.257),
                ("TotalWorkingYears", -0.244),
                ("YearsAtCompany", -0.176),
                ("MonthlyIncome", -0.170),
                ("Age", -0.159),
            ]
            fig = go.Figure(go.Bar(
                x=[abs(v) for _, v in factores_retencion],
                y=[n for n, _ in factores_retencion],
                orientation="h",
                marker_color="#34d399",
                text=[f"{v:.3f}" for _, v in factores_retencion],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=11),
            ))
            fig.update_layout(
                height=250, paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                font=dict(color="#94a3b8"), margin=dict(l=8, r=60, t=8, b=8),
                xaxis=dict(gridcolor=GRID_COLOR), yaxis=dict(gridcolor=GRID_COLOR),
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
elif seccion == "🧪 Experimentación":
    st.markdown("## 🧪 Resultados por modelo")

    modelo_sel = st.selectbox(
        "Seleccionar modelo:",
        list(RESULTADOS.keys()),
        format_func=lambda x: f"{'★ ' if x == 'Random Forest' else ''}{x}"
    )

    datos = RESULTADOS[modelo_sel]
    color_mod = MODEL_COLORS[modelo_sel]

    # Métricas principales
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{datos['accuracy']:.4f}")
    c2.metric("Precision", f"{datos['precision']:.4f}")
    c3.metric("Recall", f"{datos['recall']:.4f}")
    c4.metric("F1-Score", f"{datos['f1']:.4f}")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        # Matriz de confusión
        st.markdown("#### Matriz de Confusión")
        tp, fp, fn, tn = datos["tp"], datos["fp"], datos["fn"], datos["tn"]
        total = tp + fp + fn + tn
        cm = np.array([[tn, fp], [fn, tp]])
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        annot = [
            [f"TN={tn}<br>({cm_pct[0,0]:.1f}%)", f"FP={fp}<br>({cm_pct[0,1]:.1f}%)"],
            [f"FN={fn}<br>({cm_pct[1,0]:.1f}%)", f"TP={tp}<br>({cm_pct[1,1]:.1f}%)"],
        ]
        fig = go.Figure(go.Heatmap(
            z=cm_pct,
            x=["Pred: No (0)", "Pred: Yes (1)"],
            y=["Real: No (0)", "Real: Yes (1)"],
            colorscale=[[0, "#111827"], [0.5, "#1e3a5f"], [1, color_mod]],
            text=annot, texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
            showscale=False,
        ))
        fig.update_layout(
            height=300, paper_bgcolor=BG, plot_bgcolor=CARD_BG,
            font=dict(color="#94a3b8", family="DM Sans"),
            margin=dict(l=8, r=8, t=8, b=8),
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Radar chart de métricas
        st.markdown("#### Perfil del modelo")
        categorias = ["Accuracy", "Precision", "Recall", "F1-Score"]
        valores = [datos["accuracy"], datos["precision"], datos["recall"], datos["f1"]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=valores + [valores[0]],
            theta=categorias + [categorias[0]],
            fill="toself",
            line_color=color_mod,
            fillcolor=color_mod + "22",
            name=modelo_sel,
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_COLOR,
                                tickfont=dict(color="#64748b"), linecolor=GRID_COLOR),
                angularaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                                 tickfont=dict(color="#94a3b8")),
                bgcolor=CARD_BG,
            ),
            paper_bgcolor=BG, showlegend=False,
            height=300, margin=dict(l=16, r=16, t=16, b=16),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Descripción del modelo
    st.markdown("---")
    descripciones = {
        "SVM": "**SVM Lineal** separa las clases mediante hiperplanos de margen máximo. Con kernel lineal, es eficiente en alta dimensionalidad. Alto accuracy pero bajo recall para la clase minoritaria.",
        "KNN": "**KNN (k=5)** clasifica por vecindad. Sensible a la escala (por eso se normalizó) y al desbalance de clases. Resultó el modelo más débil en F1.",
        "Regresión Logística": "**Regresión Logística** modela la probabilidad de renuncia. Alta interpretabilidad y buenos resultados generales. Buena precision pero bajo recall.",
        "Árbol de Decisión": "**Árbol de Decisión (max_depth=5)** aprende reglas interpretables. Balance razonable entre precision y recall. Tendencia al overfitting si no se controla la profundidad.",
        "Random Forest": "**Random Forest** ensemble de 100 árboles con profundidad máx. 10. Mejor accuracy y precision general. Sin embargo, recall bajo: prioriza no generar falsos positivos.",
        "Naive Bayes": "**Naive Bayes Gaussiano** asume independencia entre features. Recall más alto de todos: detecta más renuncias reales, pero con muchos falsos positivos.",
        "Gamma (PYDRA)": "**Clasificador Gamma** usa operador de similitud con soporte a valores perdidos (tabla PYDRA variante 7). Funcionalmente correcto pero computacionalmente costoso.",
        "Perceptrón": "**Perceptrón** (l2, α=0.01) clasificador lineal de una capa. Desempeño inconsistente por sensibilidad a inicialización y datos desbalanceados.",
    }
    st.markdown(f"""
    <div class='card card-accent'>
        <p style='color:#e2e8f0; font-size:14px; line-height:1.7; margin:0;'>
            {descripciones[modelo_sel]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Métricas de confusión detalladas
    st.markdown("#### Desglose de predicciones")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Verdaderos Positivos", tp, f"{tp/(tp+fn)*100:.1f}% de renuncias")
    c2.metric("✅ Verdaderos Negativos", tn, f"{tn/(tn+fp)*100:.1f}% de retenciones")
    c3.metric("❌ Falsos Positivos", fp, "Alarmas falsas")
    c4.metric("❌ Falsos Negativos", fn, "Renuncias no detectadas")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARATIVA
# ═══════════════════════════════════════════════════════════════════════════════
elif seccion == "📈 Comparativa de Modelos":
    st.markdown("## 📈 Comparativa de todos los modelos")

    modelos = list(RESULTADOS.keys())
    colores = [MODEL_COLORS[m] for m in modelos]
    df_comp = pd.DataFrame([
        {
            "Modelo": m,
            "Accuracy": RESULTADOS[m]["accuracy"],
            "Precision": RESULTADOS[m]["precision"],
            "Recall": RESULTADOS[m]["recall"],
            "F1-Score": RESULTADOS[m]["f1"],
        }
        for m in modelos
    ]).sort_values("F1-Score", ascending=False).reset_index(drop=True)

    # ── Tabla visual
    st.markdown("#### Tabla de métricas (ordenada por F1-Score)")

    def color_val(val, col_name):
        vals = df_comp[col_name]
        maxv, minv = vals.max(), vals.min()
        if val == maxv:
            return "🟢"
        elif val == minv:
            return "🔴"
        return "⚪"

    for _, row in df_comp.iterrows():
        m = row["Modelo"]
        color = MODEL_COLORS[m]
        rank = df_comp.index[df_comp["Modelo"] == m][0] + 1
        st.markdown(f"""
        <div class='card' style='padding:12px 16px; margin-bottom:4px;'>
            <div style='display:flex; align-items:center; gap:12px;'>
                <span style='font-family:Space Mono; color:#475569; font-size:11px; min-width:24px;'>#{rank}</span>
                <span style='color:{color}; font-weight:600; font-size:13px; min-width:180px;'>■ {m}</span>
                <div style='display:flex; gap:24px; flex:1;'>
                    <div style='text-align:center; min-width:90px;'>
                        <div style='font-family:Space Mono; color:#7dd3fc; font-size:13px;'>{row["Accuracy"]:.4f}</div>
                        <div style='color:#475569; font-size:10px;'>Accuracy</div>
                    </div>
                    <div style='text-align:center; min-width:90px;'>
                        <div style='font-family:Space Mono; color:#a78bfa; font-size:13px;'>{row["Precision"]:.4f}</div>
                        <div style='color:#475569; font-size:10px;'>Precision</div>
                    </div>
                    <div style='text-align:center; min-width:90px;'>
                        <div style='font-family:Space Mono; color:#fbbf24; font-size:13px;'>{row["Recall"]:.4f}</div>
                        <div style='color:#475569; font-size:10px;'>Recall</div>
                    </div>
                    <div style='text-align:center; min-width:90px;'>
                        <div style='font-family:Space Mono; color:#34d399; font-size:13px; font-weight:700;'>{row["F1-Score"]:.4f}</div>
                        <div style='color:#475569; font-size:10px;'>F1-Score ★</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Gráficas comparativas
    metrica_sel = st.selectbox("Métrica para comparar:", ["Accuracy", "Precision", "Recall", "F1-Score"])

    col1, col2 = st.columns(2)
    with col1:
        # Bar chart de métrica seleccionada
        df_sorted = df_comp.sort_values(metrica_sel, ascending=True)
        fig = go.Figure(go.Bar(
            x=df_sorted[metrica_sel],
            y=df_sorted["Modelo"],
            orientation="h",
            marker_color=[MODEL_COLORS[m] for m in df_sorted["Modelo"]],
            text=[f"{v:.4f}" for v in df_sorted[metrica_sel]],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11, family="Space Mono"),
        ))
        fig.update_layout(
            title=f"{metrica_sel} por modelo",
            height=350, paper_bgcolor=BG, plot_bgcolor=CARD_BG,
            font=dict(color="#94a3b8", family="DM Sans"),
            margin=dict(l=8, r=60, t=48, b=8),
            xaxis=dict(gridcolor=GRID_COLOR, range=[0, 1]),
            yaxis=dict(gridcolor=GRID_COLOR),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Scatter Precision vs Recall
        fig = go.Figure()
        for m in modelos:
            d = RESULTADOS[m]
            fig.add_trace(go.Scatter(
                x=[d["recall"]], y=[d["precision"]],
                mode="markers+text",
                name=m,
                marker=dict(size=14, color=MODEL_COLORS[m], line=dict(color="white", width=1)),
                text=[m], textposition="top center",
                textfont=dict(size=9, color="#94a3b8"),
            ))
        # Línea iso-F1
        r_range = np.linspace(0.01, 1, 100)
        for f1_val in [0.2, 0.3, 0.4]:
            p_iso = f1_val * r_range / (2 * r_range - f1_val)
            p_iso = np.clip(p_iso, 0, 1)
            fig.add_trace(go.Scatter(
                x=r_range, y=p_iso, mode="lines",
                line=dict(color="#1e293b", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))
        fig.update_layout(
            title="Precision vs Recall (curvas iso-F1 punteadas)",
            height=350, paper_bgcolor=BG, plot_bgcolor=CARD_BG,
            font=dict(color="#94a3b8", family="DM Sans"),
            legend=dict(bgcolor=CARD_BG, font=dict(size=9)),
            margin=dict(l=8, r=8, t=48, b=8),
            xaxis=dict(title="Recall", gridcolor=GRID_COLOR, range=[0, 1]),
            yaxis=dict(title="Precision", gridcolor=GRID_COLOR, range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap todas las métricas
    st.markdown("#### Heatmap comparativo")
    metricas = ["Accuracy", "Precision", "Recall", "F1-Score"]
    z = [[RESULTADOS[m][met.lower().replace("-score", "").replace(" ", "")] 
          if met.lower().replace("-score", "").replace(" ", "") in RESULTADOS[m]
          else RESULTADOS[m]["f1"] if "f1" in met.lower()
          else RESULTADOS[m]["accuracy"]
          for met in metricas] for m in modelos]
    
    # Reconstruir correctamente
    z = []
    for m in modelos:
        d = RESULTADOS[m]
        z.append([d["accuracy"], d["precision"], d["recall"], d["f1"]])

    text_z = [[f"{v:.4f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=metricas, y=modelos,
        colorscale=[[0, "#1e293b"], [0.5, "#1e3a5f"], [1, "#7dd3fc"]],
        text=text_z, texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#94a3b8"),
            bgcolor=CARD_BG,
            bordercolor=GRID_COLOR,
        ),
    ))
    fig.update_layout(
        height=380, paper_bgcolor=BG, plot_bgcolor=CARD_BG,
        font=dict(color="#94a3b8", family="DM Sans"),
        margin=dict(l=8, r=8, t=16, b=8),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS CRÍTICO
# ═══════════════════════════════════════════════════════════════════════════════
elif seccion == "🔬 Análisis Crítico":
    st.markdown("## 🔬 Análisis Crítico & Conclusiones")

    # ── Hallazgos clave
    st.markdown("### 📌 Hallazgos clave")
    hallazgos = [
        ("🏆", "Random Forest — Mejor accuracy y precision",
         "Con 87.41% de accuracy y 71.43% de precision, Random Forest domina en evitar falsas alarmas. Sin embargo, su recall (21.74%) indica que aún pierde ~36 renuncias reales de 46.",
         "card-success"),
        ("📊", "Naive Bayes — Mejor recall (60.87%)",
         "Detecta la mayor proporción de renuncias reales (28 de 46), pero genera 54 falsas alarmas. Su F1 de 43.75% es el más alto, lo que lo hace valioso si el costo de perder una renuncia es alto.",
         "card-warning"),
        ("⚠️", "Desbalance de clases — El gran reto",
         "Con solo el 16.1% de empleados que renuncian, la mayoría de modelos aprenden a predecir 'No' casi siempre. Esto infla el accuracy pero perjudica el recall de la clase positiva.",
         "card-danger"),
        ("🔬", "Gamma PYDRA — Correcto pero costoso",
         "El clasificador por similitud Gamma funciona correctamente y tolera valores perdidos, pero su complejidad O(n²) lo hace inviable para datasets grandes.",
         "card-accent"),
        ("💡", "Data Leakage identificado",
         "El notebook reconoce que imputer y scaler se ajustaron sobre todo el dataset antes del split. Esto es una buena práctica de honestidad académica pero introduce sesgo optimista en las métricas.",
         "card-warning"),
    ]
    for icon, titulo, texto, card_class in hallazgos:
        st.markdown(f"""
        <div class='card {card_class}'>
            <div style='font-weight:600; color:#e2e8f0; margin-bottom:6px;'>{icon} {titulo}</div>
            <div style='color:#94a3b8; font-size:13px; line-height:1.6;'>{texto}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Recomendación final")
        st.markdown("""
        <div class='card' style='border-left: 3px solid #f87171; padding:20px;'>
            <p style='color:#7dd3fc; font-family:Space Mono; font-size:12px; margin-bottom:12px;'>MODELO RECOMENDADO</p>
            <h3 style='color:#f8fafc; font-size:1.4rem; margin:0 0 8px 0;'>🌲 Random Forest</h3>
            <p style='color:#94a3b8; font-size:13px; line-height:1.7;'>
                Para un contexto de RRHH donde identificar empleados en riesgo de renuncia 
                es prioritario, se recomienda <b>Random Forest</b> como modelo principal, 
                complementado con un <b>ajuste del umbral de decisión</b> para mejorar el 
                recall sin sacrificar demasiada precisión.
            </p>
            <div style='margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;'>
                <span class='badge badge-blue'>Alta precisión</span>
                <span class='badge badge-green'>Robusto</span>
                <span class='badge badge-yellow'>Ajustable</span>
                <span class='badge badge-red'>Mejor F1</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🚀 Mejoras propuestas")
        mejoras = [
            "Aplicar SMOTE para balancear clases en entrenamiento",
            "Ajustar umbral de decisión (p.ej. 0.3 en lugar de 0.5)",
            "Corrección del data leakage (fit solo en X_train)",
            "Cross-validation estratificada (k-fold)",
            "Búsqueda de hiperparámetros con GridSearchCV",
            "Agregar métricas AUC-ROC y curva Precision-Recall",
        ]
        for i, m in enumerate(mejoras, 1):
            st.markdown(f"""
            <div class='step'>
                <span class='step-num'>{i:02d}</span>
                <span style='color:#cbd5e1; font-size:13px;'>{m}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### ⚖️ Trade-off Precision vs Recall")
        st.markdown("""
        <div class='card' style='margin-bottom:12px;'>
            <p style='color:#94a3b8; font-size:13px; line-height:1.7; margin:0;'>
                En RRHH, el costo de un <b style='color:#f87171'>Falso Negativo</b> 
                (no detectar una renuncia real) suele ser mayor que el de un 
                <b style='color:#fbbf24'>Falso Positivo</b> (intervenir innecesariamente). 
                Por ello, <b>aumentar el Recall</b> tiene valor estratégico.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Comparativa visual simplificada
        metricas_comp = ["Accuracy", "Precision", "Recall", "F1-Score"]
        rf_vals = [0.8741, 0.7143, 0.2174, 0.3333]
        nb_vals = [0.7891, 0.3415, 0.6087, 0.4375]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Random Forest", x=metricas_comp, y=rf_vals,
                             marker_color="#f87171", text=[f"{v:.3f}" for v in rf_vals],
                             textposition="outside", textfont=dict(color="#94a3b8", size=11)))
        fig.add_trace(go.Bar(name="Naive Bayes", x=metricas_comp, y=nb_vals,
                             marker_color="#fb923c", text=[f"{v:.3f}" for v in nb_vals],
                             textposition="outside", textfont=dict(color="#94a3b8", size=11)))
        fig.update_layout(
            title="RF vs Naive Bayes: el trade-off clave",
            barmode="group", height=300,
            paper_bgcolor=BG, plot_bgcolor=CARD_BG,
            font=dict(color="#94a3b8", family="DM Sans"),
            legend=dict(bgcolor=CARD_BG, font=dict(size=11)),
            margin=dict(l=8, r=8, t=48, b=8),
            yaxis=dict(gridcolor=GRID_COLOR, range=[0, 1]),
            xaxis=dict(gridcolor=GRID_COLOR),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📝 Conclusión del equipo")
        st.markdown("""
        <div class='card card-accent' style='padding:20px;'>
            <p style='color:#94a3b8; font-size:13px; line-height:1.8; margin:0;'>
            De los 8 clasificadores evaluados, <b style='color:#f8fafc'>Random Forest</b> 
            obtuvo el mejor desempeño general. <b style='color:#f8fafc'>SVM</b> y 
            <b style='color:#f8fafc'>Regresión Logística</b> son alternativas sólidas 
            e interpretables. <b style='color:#f8fafc'>Naive Bayes</b> destaca en recall, 
            útil si la prioridad es no perder renuncias. El clasificador 
            <b style='color:#f8fafc'>Gamma PYDRA</b> es funcionalmente correcto pero 
            computacionalmente inviable a escala. <b style='color:#f8fafc'>KNN</b> y 
            <b style='color:#f8fafc'>Perceptrón</b> resultaron los más débiles.
            </p>
        </div>
        """, unsafe_allow_html=True)
