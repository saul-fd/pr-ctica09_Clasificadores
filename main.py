"""
Dashboard Práctica 09 - Clasificadores de Rotación de Personal
Equipo 12
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Práctica 09 · Clasificadores",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0b0f1a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1e293b; }
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
h1 { font-family: 'Space Mono', monospace; color: #f8fafc !important; letter-spacing: -1px; }
h2 { font-family: 'Space Mono', monospace; color: #7dd3fc !important; font-size: 1.15rem !important; }
[data-testid="metric-container"] { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 16px; }
[data-testid="metric-container"] label { color: #64748b !important; font-size: 12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #7dd3fc !important; font-family: 'Space Mono', monospace; }
.stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; border-radius: 8px; font-size: 13px; }
.stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #7dd3fc !important; }
hr { border-color: #1e293b !important; }
.card { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 10px; }
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
    st.markdown("### 🤖 Práctica 09")
    st.markdown("**Clasificadores · Equipo 12**")
    st.markdown("---")
    seccion = st.radio("", [
        "🏠 Resumen general",
        "🧪 Explorar modelo",
        "📈 Comparativa",
        "🔬 Análisis crítico",
    ])
    st.markdown("---")
    st.markdown("""<div style='font-size:12px;color:#475569;'>
        <b>Equipo 12</b><br>
        Saul Fabila Domínguez<br>
        Sergio Iván Sánchez Portilla<br>
        Diana María Vélez Gallardo
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 🏠 RESUMEN GENERAL
# ══════════════════════════════════════════════════════════════════
if seccion == "🏠 Resumen general":
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #1e293b;border-radius:16px;padding:28px;margin-bottom:20px;'>
        <p style='font-family:Space Mono;font-size:10px;color:#7dd3fc;letter-spacing:3px;margin:0 0 6px;'>PRÁCTICA 09 · EQUIPO 12</p>
        <h1 style='font-size:1.8rem;margin:0 0 6px;'>Clasificadores de Rotación de Personal</h1>
        <p style='color:#64748b;font-size:13px;margin:0;'>IBM HR Analytics · 8 modelos · Predicción de Attrition</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", "1,470 registros", "IBM HR Analytics")
    c2.metric("Modelos evaluados", "8")
    c3.metric("Mejor Accuracy", "87.41%", "Random Forest")
    c4.metric("Mejor F1-Score", "43.75%", "Naive Bayes")

    st.markdown("---")
    st.markdown("## Distribución de clases (Attrition)")

    clase_df = pd.DataFrame({
        "Clase": ["No renuncia (0)", "Renuncia (1)"],
        "Cantidad": [1233, 237],
    })
    st.bar_chart(clase_df.set_index("Clase"), height=280, color="#7dd3fc")

    st.markdown("""<div class='card' style='border-left:3px solid #fbbf24; margin-top:8px;'>
        ⚠️ <b>Desbalance severo: 83.9% vs 16.1%</b> — La mayoría de modelos aprenden a predecir
        "No renuncia" casi siempre, inflando el Accuracy pero perjudicando el Recall de la clase positiva.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Ranking de modelos por F1-Score")
    df_rank = pd.DataFrame([
        {"Modelo": m, "Accuracy": d["accuracy"], "Precision": d["precision"],
         "Recall": d["recall"], "F1-Score": d["f1"]}
        for m, d in RESULTADOS.items()
    ]).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    df_rank.index += 1

    st.bar_chart(
        df_rank.set_index("Modelo")[["F1-Score"]],
        height=300,
        color="#34d399",
    )
    st.dataframe(
        df_rank.style.format({"Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}","F1-Score":"{:.4f}"})
            .highlight_max(axis=0, color="#14432a")
            .highlight_min(axis=0, color="#3d1010"),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════
# 🧪 EXPLORAR MODELO
# ══════════════════════════════════════════════════════════════════
elif seccion == "🧪 Explorar modelo":
    st.markdown("## 🧪 Explorar modelo")

    modelo_sel = st.selectbox(
        "Selecciona un modelo:",
        list(RESULTADOS.keys()),
        format_func=lambda x: f"★ {x}" if x == "Random Forest" else x,
    )
    d = RESULTADOS[modelo_sel]
    st.markdown("---")

    # Métricas
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
            "Métrica":  ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Valor":    [d["accuracy"], d["precision"], d["recall"], d["f1"]],
        }).set_index("Métrica")
        st.bar_chart(perfil_df, height=300, color="#7dd3fc")

    with col2:
        st.markdown("#### Matriz de confusión")
        tp, fp, fn, tn = d["tp"], d["fp"], d["fn"], d["tn"]
        total = tp + fp + fn + tn

        # Tabla con emojis para hacerla visual
        cm_df = pd.DataFrame(
            {
                "Pred: No (0)": [f"✅ TN = {tn}  ({tn/total*100:.1f}%)", f"❌ FN = {fn}  ({fn/total*100:.1f}%)"],
                "Pred: Yes (1)": [f"❌ FP = {fp}  ({fp/total*100:.1f}%)", f"✅ TP = {tp}  ({tp/total*100:.1f}%)"],
            },
            index=["Real: No (0)", "Real: Yes (1)"],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("#### Desglose de predicciones")
        breakdown_df = pd.DataFrame({
            "Categoría": ["Verdadero Positivo (TP)", "Verdadero Negativo (TN)", "Falso Positivo (FP)", "Falso Negativo (FN)"],
            "Cantidad":  [tp, tn, fp, fn],
        }).set_index("Categoría")
        st.bar_chart(breakdown_df, height=220, color=["#34d399", "#34d399", "#f87171", "#f87171"])

    st.markdown("---")

    # Comparación con todos los modelos en esa métrica
    st.markdown("#### ¿Cómo se compara este modelo en cada métrica?")
    tabs = st.tabs(["Accuracy", "Precision", "Recall", "F1-Score"])
    for tab, met in zip(tabs, ["accuracy", "precision", "recall", "f1"]):
        with tab:
            df_met = pd.DataFrame({
                "Modelo": list(RESULTADOS.keys()),
                "Valor":  [RESULTADOS[m][met] for m in RESULTADOS],
            }).sort_values("Valor", ascending=False).set_index("Modelo")
            # Resaltar el seleccionado
            st.bar_chart(df_met, height=270, color="#7dd3fc")
            val_sel = RESULTADOS[modelo_sel][met]
            rank = sorted([RESULTADOS[m][met] for m in RESULTADOS], reverse=True).index(val_sel) + 1
            st.caption(f"**{modelo_sel}** ocupa el puesto **#{rank}** con {val_sel:.4f}")


# ══════════════════════════════════════════════════════════════════
# 📈 COMPARATIVA
# ══════════════════════════════════════════════════════════════════
elif seccion == "📈 Comparativa":
    st.markdown("## 📈 Comparativa de modelos")

    modelos = list(RESULTADOS.keys())
    df_all = pd.DataFrame([
        {"Modelo": m, "Accuracy": d["accuracy"], "Precision": d["precision"],
         "Recall": d["recall"], "F1-Score": d["f1"]}
        for m, d in RESULTADOS.items()
    ]).set_index("Modelo")

    # Tabla con highlight
    st.markdown("#### Tabla comparativa")
    st.dataframe(
        df_all.sort_values("F1-Score", ascending=False)
            .style.format("{:.4f}")
            .highlight_max(axis=0, color="#14432a")
            .highlight_min(axis=0, color="#3d1010"),
        use_container_width=True,
    )
    st.markdown("---")

    # Gráficas por métrica en 2x2
    st.markdown("#### Comparativa por métrica")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Accuracy**")
        st.bar_chart(df_all[["Accuracy"]].sort_values("Accuracy"), height=260, color="#7dd3fc")
        st.markdown("**Recall**")
        st.bar_chart(df_all[["Recall"]].sort_values("Recall"), height=260, color="#fbbf24")

    with col2:
        st.markdown("**Precision**")
        st.bar_chart(df_all[["Precision"]].sort_values("Precision"), height=260, color="#a78bfa")
        st.markdown("**F1-Score**")
        st.bar_chart(df_all[["F1-Score"]].sort_values("F1-Score"), height=260, color="#34d399")

    st.markdown("---")

    # Multi-line chart: todos los modelos, todas las métricas
    st.markdown("#### Todas las métricas juntas por modelo")
    st.line_chart(df_all, height=350)

    # Selector de modelos para comparar 2
    st.markdown("---")
    st.markdown("#### Comparar dos modelos directamente")
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
    st.bar_chart(df_duo, height=320)

    # Diferencias
    st.markdown("#### Diferencia (A − B)")
    diff_df = pd.DataFrame({
        "Métrica": labels,
        "Diferencia": [round(da[m]-db[m], 4) for m in metricas],
    }).set_index("Métrica")
    st.dataframe(diff_df.style.format("{:.4f}")
        .applymap(lambda v: "color: #34d399" if v > 0 else "color: #f87171"),
        use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 🔬 ANÁLISIS CRÍTICO
# ══════════════════════════════════════════════════════════════════
elif seccion == "🔬 Análisis crítico":
    st.markdown("## 🔬 Análisis crítico & conclusiones")

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("### Hallazgos clave")
        hallazgos = [
            ("🏆", "#34d399", "Random Forest — mejor accuracy y precision",
             "87.41% de accuracy y 71.43% de precision. Evita falsas alarmas pero pierde ~36 renuncias reales (recall bajo)."),
            ("📊", "#fb923c", "Naive Bayes — mejor recall y F1-Score",
             "60.87% de recall y F1 de 43.75%. Detecta más renuncias reales a costo de 54 falsas alarmas."),
            ("⚖️", "#f87171", "Desbalance de clases es el principal reto",
             "83.9% vs 16.1%. Infla el Accuracy y penaliza el Recall. Se recomienda SMOTE o ajuste de umbral."),
            ("🔬", "#2dd4bf", "Gamma PYDRA — correcto pero O(n²)",
             "Funcional y tolerante a valores perdidos, pero computacionalmente inviable para datasets grandes."),
            ("💡", "#fbbf24", "Data Leakage identificado y documentado",
             "Imputer y Scaler se ajustaron antes del split. Las métricas pueden ser ligeramente optimistas."),
        ]
        for icon, color, titulo, texto in hallazgos:
            st.markdown(f"""<div style='background:#111827;border:1px solid #1e293b;
                border-left:3px solid {color};border-radius:10px;padding:14px;margin-bottom:8px;'>
                <b style='color:#e2e8f0;'>{icon} {titulo}</b><br>
                <span style='color:#94a3b8;font-size:13px;'>{texto}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### Trade-off Precision vs Recall")
        tradeoff_df = pd.DataFrame({
            "Precision": [d["precision"] for d in RESULTADOS.values()],
            "Recall":    [d["recall"] for d in RESULTADOS.values()],
        }, index=RESULTADOS.keys())
        st.scatter_chart(tradeoff_df, x="Recall", y="Precision", size=80, height=300)
        st.caption("Cada punto es un modelo. Arriba-derecha = ideal. Naive Bayes tiene alto Recall; Random Forest alta Precision.")

        st.markdown("### RF vs Naive Bayes")
        rf, nb = RESULTADOS["Random Forest"], RESULTADOS["Naive Bayes"]
        df_vs = pd.DataFrame({
            "Random Forest": [rf["accuracy"], rf["precision"], rf["recall"], rf["f1"]],
            "Naive Bayes":   [nb["accuracy"], nb["precision"], nb["recall"], nb["f1"]],
        }, index=["Accuracy","Precision","Recall","F1-Score"])
        st.bar_chart(df_vs, height=250)

    st.markdown("---")
    st.markdown("### Mejoras propuestas")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""<div class='card' style='border-left:3px solid #7dd3fc;'>
        <b style='color:#7dd3fc;'>Datos</b><br>
        <span style='font-size:13px;color:#94a3b8;'>
        · Aplicar SMOTE para balancear<br>
        · Corregir data leakage<br>
        · Validación cruzada k-fold
        </span>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""<div class='card' style='border-left:3px solid #34d399;'>
        <b style='color:#34d399;'>Modelos</b><br>
        <span style='font-size:13px;color:#94a3b8;'>
        · GridSearchCV para hiperparámetros<br>
        · Ajustar umbral a 0.30<br>
        · Probar XGBoost / LightGBM
        </span>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""<div class='card' style='border-left:3px solid #a78bfa;'>
        <b style='color:#a78bfa;'>Evaluación</b><br>
        <span style='font-size:13px;color:#94a3b8;'>
        · Agregar AUC-ROC<br>
        · Curva Precision-Recall<br>
        · Análisis de importancia de features
        </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style='background:#111827;border:1px solid #1e293b;border-left:3px solid #7dd3fc;
        border-radius:10px;padding:20px;'>
        <p style='font-family:Space Mono;font-size:10px;color:#7dd3fc;letter-spacing:2px;margin:0 0 8px;'>CONCLUSIÓN EQUIPO 12</p>
        <p style='color:#cbd5e1;font-size:14px;line-height:1.8;margin:0;'>
        <b>Random Forest</b> es el modelo recomendado como punto de partida por su alta precision.
        <b>Naive Bayes</b> es preferible si detectar renuncias reales es prioritario sobre evitar falsas alarmas.
        El mayor reto fue el <b>desbalance de clases</b>, que limitó el Recall de todos los modelos.
        <b>KNN</b> y <b>Perceptrón</b> resultaron los más débiles. El clasificador <b>Gamma PYDRA</b>
        demostró valor teórico pero es inviable en producción por su complejidad computacional.
        </p>
    </div>""", unsafe_allow_html=True)
