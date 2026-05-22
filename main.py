"""
Dashboard Práctica 09 - Clasificadores de Rotación de Personal
Equipo 12 | Solo usa: streamlit, pandas, numpy
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Práctica 09 · Clasificadores",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #ffffff; color: #1e3a5f; }
section[data-testid="stSidebar"] { background-color: #daeaf7; border-right: 1px solid #b8d6ee; }
section[data-testid="stSidebar"] * { color: #2c5f8a !important; }
h1 { font-family: 'Space Mono', monospace; color: #1e3a5f !important; letter-spacing: -1px; }
h2 { font-family: 'Space Mono', monospace; color: #2e7cbf !important; font-size: 1.15rem !important; }
[data-testid="metric-container"] { background: #f0f7fd; border: 1px solid #b8d6ee; border-radius: 12px; padding: 16px; }
[data-testid="metric-container"] label { color: #5a8ab0 !important; font-size: 12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #1a5d96 !important; font-family: 'Space Mono', monospace; }
.stTabs [data-baseweb="tab-list"] { background: #e8f3fb; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #5a8ab0; border-radius: 8px; font-size: 13px; }
.stTabs [aria-selected="true"] { background: #8abfdf !important; color: #0d2d4e !important; }
hr { border-color: #b8d6ee !important; }
.card { background: #f0f7fd; border: 1px solid #b8d6ee; border-radius: 12px; padding: 16px; margin-bottom: 10px; }
/* Texto general */
p, span, div, li { color: #1e3a5f; }
/* Dataframes */
[data-testid="stDataFrame"] { background: #f0f7fd !important; }
</style>
""", unsafe_allow_html=True)

# ── DATOS ─────────────────────────────────────────────────────────────────────
RESULTADOS = {
    "Random Forest":       {"accuracy":0.8741,"precision":0.7143,"recall":0.2174,"f1":0.3333,"tp":10,"fp":4, "fn":36,"tn":246},
    "Naive Bayes":         {"accuracy":0.7891,"precision":0.3415,"recall":0.6087,"f1":0.4375,"tp":28,"fp":54,"fn":18,"tn":196},
    "Regresión Logística": {"accuracy":0.8707,"precision":0.6250,"recall":0.2174,"f1":0.3226,"tp":10,"fp":6, "fn":36,"tn":244},
    "SVM":                 {"accuracy":0.8605,"precision":0.5556,"recall":0.2174,"f1":0.3125,"tp":10,"fp":8, "fn":36,"tn":242},
    "Árbol de Decisión":   {"accuracy":0.8435,"precision":0.4138,"recall":0.2609,"f1":0.3200,"tp":12,"fp":17,"fn":34,"tn":233},
    "Gamma (PYDRA)":       {"accuracy":0.8435,"precision":0.4286,"recall":0.1957,"f1":0.2687,"tp":9, "fp":12,"fn":37,"tn":238},
    "Perceptrón":          {"accuracy":0.7925,"precision":0.2593,"recall":0.3043,"f1":0.2800,"tp":14,"fp":40,"fn":32,"tn":210},
    "KNN":                 {"accuracy":0.8299,"precision":0.3684,"recall":0.1522,"f1":0.2157,"tp":7, "fp":12,"fn":39,"tn":238},
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Práctica 09")
    st.markdown("**Clasificadores · Equipo 12**")
    st.markdown("---")
    seccion = st.radio("", [
        "Resumen general",
        "Explorar modelo",
        "Comparativa",
    ])
    st.markdown("---")
    st.markdown("""<div style='font-size:12px;color:#5a8ab0;'>
        <b>Equipo 12</b><br>
        Saul Fabila Domínguez<br>
        Sergio Iván Sánchez Portilla<br>
        Diana María Vélez Gallardo
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# RESUMEN GENERAL
# ══════════════════════════════════════════════════════════════════
if seccion == "Resumen general":
    st.markdown("""
    <div style='background:linear-gradient(135deg,#daeaf7,#eef6fc);border:1px solid #b8d6ee;border-radius:16px;padding:28px;margin-bottom:20px;'>
        <p style='font-family:Space Mono;font-size:10px;color:#2e7cbf;letter-spacing:3px;margin:0 0 6px;'>PRÁCTICA 09 · EQUIPO 12</p>
        <h1 style='font-size:1.8rem;margin:0 0 6px;color:#1e3a5f !important;'>Clasificadores de Rotación de Personal</h1>
        <p style='color:#5a8ab0;font-size:13px;margin:0;'>IBM HR Analytics · 8 modelos · Predicción de Attrition</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", "1,470 registros", "IBM HR Analytics")
    c2.metric("Modelos evaluados", "8")
    c3.metric("Mejor Accuracy", "87.41%", "Random Forest")
    c4.metric("Mejor F1-Score", "43.75%", "Naive Bayes")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("## Distribución de clases (Attrition)")
        clase_df = pd.DataFrame({
            "Clase": ["No renuncia (0)", "Renuncia (1)"],
            "Cantidad": [1233, 237],
        }).set_index("Clase")
        st.bar_chart(clase_df, height=260, color="#5ba3d9")
        st.markdown("""<div class='card' style='border-left:3px solid #f4a94e;'>
            <b>Desbalance: 83.9% vs 16.1%</b><br>
            <span style='color:#5a8ab0;font-size:13px;'>La mayoría de modelos aprenden a predecir
            </span>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("## Ranking por F1-Score")
        df_rank = pd.DataFrame([
            {"Modelo": m, "Accuracy": d["accuracy"], "Precision": d["precision"],
             "Recall": d["recall"], "F1-Score": d["f1"]}
            for m, d in RESULTADOS.items()
        ]).sort_values("F1-Score", ascending=False).reset_index(drop=True)
        df_rank.index += 1
        st.bar_chart(df_rank.set_index("Modelo")[["F1-Score"]], height=260, color="#7ec8a4")
        st.dataframe(
            df_rank.style
                .format({"Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}","F1-Score":"{:.4f}"})
                .highlight_max(axis=0, color="#b7e4cc")
                .highlight_min(axis=0, color="#f7c5c5"),
            use_container_width=True,
            height=320,
        )


# ══════════════════════════════════════════════════════════════════
# EXPLORAR MODELO
# ══════════════════════════════════════════════════════════════════
elif seccion == "Explorar modelo":
    st.markdown("## Explorar modelo")

    modelo_sel = st.selectbox(
        "Selecciona un modelo:",
        list(RESULTADOS.keys()),
    )
    d = RESULTADOS[modelo_sel]
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{d['accuracy']:.4f}")
    c2.metric("Precision", f"{d['precision']:.4f}")
    c3.metric("Recall",    f"{d['recall']:.4f}")
    c4.metric("F1-Score",  f"{d['f1']:.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Perfil de métricas")
        perfil_df = pd.DataFrame({
            "Métrica": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Valor":   [d["accuracy"], d["precision"], d["recall"], d["f1"]],
        }).set_index("Métrica")
        st.bar_chart(perfil_df, height=300, color="#5ba3d9")

        st.markdown("#### Desglose de predicciones")
        tp, fp, fn, tn = d["tp"], d["fp"], d["fn"], d["tn"]
        breakdown_df = pd.DataFrame({
            "Categoria": ["Verdadero Positivo (TP)", "Verdadero Negativo (TN)", "Falso Positivo (FP)", "Falso Negativo (FN)"],
            "Cantidad":  [tp, tn, fp, fn],
        }).set_index("Categoria")
        st.bar_chart(breakdown_df, height=260, color="#5ba3d9")

    with col2:
        st.markdown("#### Matriz de confusion")
        total = tp + fp + fn + tn
        cm_df = pd.DataFrame(
            {
                "Pred: No (0)": [f"TN = {tn}  ({tn/total*100:.1f}%)", f"FN = {fn}  ({fn/total*100:.1f}%)"],
                "Pred: Yes (1)": [f"FP = {fp}  ({fp/total*100:.1f}%)", f"TP = {tp}  ({tp/total*100:.1f}%)"],
            },
            index=["Real: No (0)", "Real: Yes (1)"],
        )
        st.dataframe(cm_df, use_container_width=True, height=105)

        st.markdown("#### Comparativa por metrica")
        tabs = st.tabs(["Accuracy", "Precision", "Recall", "F1-Score"])
        for tab, met in zip(tabs, ["accuracy", "precision", "recall", "f1"]):
            with tab:
                df_met = pd.DataFrame({
                    "Modelo": list(RESULTADOS.keys()),
                    "Valor":  [RESULTADOS[m][met] for m in RESULTADOS],
                }).sort_values("Valor", ascending=False).set_index("Modelo")
                st.bar_chart(df_met, height=250, color="#5ba3d9")
                val_sel = RESULTADOS[modelo_sel][met]
                rank = sorted([RESULTADOS[m][met] for m in RESULTADOS], reverse=True).index(val_sel) + 1
                st.caption(f"{modelo_sel} ocupa el puesto #{rank} con {val_sel:.4f}")


# ══════════════════════════════════════════════════════════════════
# COMPARATIVA
# ══════════════════════════════════════════════════════════════════
elif seccion == "Comparativa":
    st.markdown("## Comparativa de modelos")

    modelos = list(RESULTADOS.keys())
    df_all = pd.DataFrame([
        {"Modelo": m, "Accuracy": d["accuracy"], "Precision": d["precision"],
         "Recall": d["recall"], "F1-Score": d["f1"]}
        for m, d in RESULTADOS.items()
    ]).set_index("Modelo")

    st.markdown("#### Tabla comparativa")
    st.dataframe(
        df_all.sort_values("F1-Score", ascending=False)
            .style.format("{:.4f}")
            .highlight_max(axis=0, color="#b7e4cc")
            .highlight_min(axis=0, color="#f7c5c5"),
        use_container_width=True,
        height=330,
    )
    st.markdown("---")

    st.markdown("#### Comparativa por metrica")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Accuracy**")
        st.bar_chart(df_all[["Accuracy"]].sort_values("Accuracy"), height=240, color="#5ba3d9")
        st.markdown("**Recall**")
        st.bar_chart(df_all[["Recall"]].sort_values("Recall"), height=240, color="#f4a94e")
    with col2:
        st.markdown("**Precision**")
        st.bar_chart(df_all[["Precision"]].sort_values("Precision"), height=240, color="#b49de8")
        st.markdown("**F1-Score**")
        st.bar_chart(df_all[["F1-Score"]].sort_values("F1-Score"), height=240, color="#7ec8a4")

    st.markdown("---")
    st.markdown("#### Todas las metricas por modelo")
    st.line_chart(df_all, height=320)

    st.markdown("---")
    st.markdown("#### Comparar dos modelos")
    c1, c2 = st.columns(2)
    mod_a = c1.selectbox("Modelo A", modelos, index=0)
    mod_b = c2.selectbox("Modelo B", modelos, index=1)

    da, db = RESULTADOS[mod_a], RESULTADOS[mod_b]
    metricas = ["accuracy", "precision", "recall", "f1"]
    labels   = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df_duo = pd.DataFrame({
        mod_a: [da[m] for m in metricas],
        mod_b: [db[m] for m in metricas],
    }, index=labels)
    st.bar_chart(df_duo, height=300)

    st.markdown("#### Diferencia (A - B)")
    diff_df = pd.DataFrame({
        "Metrica": labels,
        "Diferencia": [round(da[m]-db[m], 4) for m in metricas],
    }).set_index("Metrica")
    st.dataframe(
        diff_df.style.format("{:.4f}").map(lambda v: "color: #2e7a4f" if v > 0 else "color: #c0392b"),
        use_container_width=True,
        height=180,
    )
