"""
Dashboard Práctica 09 - Clasificadores de Rotación de Personal
Equipo 12: Saul Fabila, Sergio Sánchez, Diana Vélez
Solo usa: streamlit, pandas, numpy, matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ─── CONFIG ───────────────────────────────────────────────────────────────────
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
section[data-testid="stSidebar"] { background-color: #111827; border-right:1px solid #1e293b; }
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
h1 { font-family:'Space Mono',monospace; color:#f8fafc !important; letter-spacing:-1px; }
h2 { font-family:'Space Mono',monospace; color:#7dd3fc !important; font-size:1.2rem !important; }
h3 { color:#94a3b8 !important; font-weight:500; font-size:1rem !important; }
[data-testid="metric-container"] { background:#111827; border:1px solid #1e293b; border-radius:12px; padding:16px; }
[data-testid="metric-container"] label { color:#64748b !important; font-size:12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#7dd3fc !important; font-family:'Space Mono',monospace; }
.stTabs [data-baseweb="tab-list"] { background:#111827; border-radius:10px; padding:4px; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#64748b; border-radius:8px; font-size:13px; }
.stTabs [aria-selected="true"] { background:#1e3a5f !important; color:#7dd3fc !important; }
hr { border-color:#1e293b !important; }
.card { background:#111827; border:1px solid #1e293b; border-radius:12px; padding:16px; margin-bottom:10px; }
</style>
""", unsafe_allow_html=True)

# ─── DATOS ────────────────────────────────────────────────────────────────────
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
COLORS = {
    "Random Forest":"#f87171","Naive Bayes":"#fb923c","Regresión Logística":"#34d399",
    "SVM":"#7dd3fc","Árbol de Decisión":"#fbbf24","Gamma (PYDRA)":"#2dd4bf",
    "Perceptrón":"#f472b6","KNN":"#a78bfa",
}
BG, CARD, GRID = "#0b0f1a", "#111827", "#1e293b"

def mpl_defaults(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors="#64748b")
    ax.spines[:].set_color(GRID)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_color("#64748b")
        item.set_fontsize(9)
    ax.yaxis.label.set_color("#94a3b8")
    ax.xaxis.label.set_color("#94a3b8")
    ax.title.set_color("#f8fafc")
    ax.grid(color=GRID, linewidth=0.5)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Práctica 09")
    st.markdown("**Clasificadores de Attrition**")
    st.markdown("---")
    seccion = st.radio("", ["🏠 Inicio","📊 Preprocesamiento","🧪 Resultados por modelo","📈 Comparativa","🔬 Análisis Crítico"])
    st.markdown("---")
    st.markdown("<div style='font-size:12px;color:#475569;'><b>Equipo 12</b><br>Saul Fabila Domínguez<br>Sergio Iván Sánchez Portilla<br>Diana María Vélez Gallardo</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# INICIO
# ══════════════════════════════════════════════════════════════════
if seccion == "🏠 Inicio":
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0f172a,#1e293b,#0f172a);border:1px solid #1e293b;border-radius:16px;padding:32px;margin-bottom:24px;'>
        <p style='font-family:Space Mono;font-size:11px;color:#7dd3fc;letter-spacing:2px;text-transform:uppercase;margin:0 0 8px;'>Práctica 09 · Equipo 12</p>
        <h1 style='font-family:Space Mono;font-size:1.9rem;color:#f8fafc;margin:0 0 8px;letter-spacing:-1px;'>Clasificadores de<br>Rotación de Personal</h1>
        <p style='color:#64748b;font-size:14px;margin:0;'>IBM HR Analytics · 8 modelos evaluados · Predicción de Attrition</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Registros", "1,470", "IBM HR Dataset")
    c2.metric("Modelos evaluados", "8", "clasificadores")
    c3.metric("Mejor accuracy", "87.41%", "Random Forest")
    c4.metric("Mejor F1-Score", "43.75%", "Naive Bayes")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## 🎯 Objetivo")
        st.markdown("""<div class='card' style='border-left:3px solid #7dd3fc;'>
            Predecir si un empleado va a <b style='color:#f87171'>renunciar</b> o no 
            (<code style='color:#7dd3fc'>Attrition = Yes/No</code>) usando datos de RRHH de IBM.
        </div>""", unsafe_allow_html=True)
        st.markdown("## 🤖 Modelos evaluados")
        df_rank = pd.DataFrame([{"Modelo":m,"F1-Score":d["f1"],"Accuracy":d["accuracy"]} for m,d in RESULTADOS.items()])
        df_rank = df_rank.sort_values("F1-Score",ascending=False).reset_index(drop=True)
        df_rank.index += 1
        st.dataframe(df_rank.style.format({"F1-Score":"{:.4f}","Accuracy":"{:.4f}"}), use_container_width=True)

    with col2:
        st.markdown("## ⚖️ Distribución de clases")
        fig, ax = plt.subplots(figsize=(5,3))
        bars = ax.bar(["No renuncia (0)","Renuncia (1)"],[1233,237],color=["#34d399","#f87171"],edgecolor=GRID,width=0.5)
        for b,v in zip(bars,[1233,237]):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+15, f"{v}", ha='center', color="#94a3b8", fontsize=11, fontweight='bold')
        ax.text(0, 1233/2, "83.9%", ha='center', va='center', color=BG, fontsize=13, fontweight='bold')
        ax.text(1, 237/2,  "16.1%", ha='center', va='center', color=BG, fontsize=13, fontweight='bold')
        ax.set_title("Distribución de Attrition", color="#f8fafc", fontsize=12, pad=10)
        ax.set_ylim(0, 1400)
        mpl_defaults(ax, fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""<div class='card' style='border-left:3px solid #fbbf24; margin-top:8px;'>
            ⚠️ <b>Desbalance: 84% vs 16%</b><br>
            <span style='color:#94a3b8;font-size:13px;'>El dataset está fuertemente desbalanceado.
            La mayoría de modelos tienden a predecir "No renuncia" casi siempre,
            inflando el accuracy pero perjudicando el recall.</span>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════
elif seccion == "📊 Preprocesamiento":
    st.markdown("## 📊 Dataset & Preprocesamiento")
    tab1, tab2, tab3 = st.tabs(["📁 Dataset","🔧 Limpieza & Codificación","📐 Correlaciones"])

    with tab1:
        c1,c2,c3 = st.columns(3)
        c1.metric("Total registros","1,470"); c2.metric("Features originales","35"); c3.metric("Features finales","46","+11 one-hot")
        st.markdown("---")
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("#### Variables numéricas clave")
            st.dataframe(pd.DataFrame({"Variable":["Age","MonthlyIncome","YearsAtCompany","TotalWorkingYears","DistanceFromHome","JobSatisfaction"],
                "Rango":["18-60","1009-19999","0-40","0-40","1-29","1-4"]}), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("#### Variables categóricas")
            st.dataframe(pd.DataFrame({"Variable":["Department","JobRole","MaritalStatus","BusinessTravel","EducationField","OverTime"],
                "Codificación":["One-hot","One-hot","One-hot","Ordinal (0-2)","One-hot","Binaria (0/1)"]}), use_container_width=True, hide_index=True)
        st.markdown("---")
        c1,c2 = st.columns(2)
        c1.metric("Train","1,176 muestras","80%"); c2.metric("Test","294 muestras","20% · stratify=y")

    with tab2:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("#### Valores nulos generados")
            st.dataframe(pd.DataFrame({"Columna":["TrainingTimesLastYear","YearsSinceLastPromotion","YearsWithCurrManager"],
                "Fracción":["5%","15%","12%"],"Estrategia":["Media","Media","Media"]}), use_container_width=True, hide_index=True)
            st.markdown("""<div class='card' style='border-left:3px solid #fbbf24;margin-top:8px;'>
                ⚠️ <b>Data Leakage identificado</b><br>
                <span style='color:#94a3b8;font-size:12px;'>Imputer y Scaler se ajustaron sobre todo el dataset antes del split.
                El flujo correcto: <code>fit(X_train) → transform(X_train) → transform(X_test)</code></span>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("#### Columnas eliminadas")
            st.dataframe(pd.DataFrame({"Columna":["Over18","EmployeeCount","StandardHours","EmployeeNumber"],
                "Motivo":["Varianza cero","Constante = 1","Constante = 80","ID sin valor predictivo"]}), use_container_width=True, hide_index=True)
            st.markdown("#### Codificaciones aplicadas")
            st.dataframe(pd.DataFrame({"Variable":["Attrition","Gender","OverTime","BusinessTravel"],
                "Mapeo":["Yes→1, No→0","Male→1, Female→0","Yes→1, No→0","0/1/2 ordinal"]}), use_container_width=True, hide_index=True)

    with tab3:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔴 Top 5: impulsan la RENUNCIA")
            riesgo = [("OverTime",0.246),("MaritalStatus_Single",0.175),("JobRole_Sales Rep.",0.148),("BusinessTravel",0.131),("DistanceFromHome",0.097)]
            fig,ax = plt.subplots(figsize=(5,3))
            nombres = [r[0] for r in riesgo]; vals = [r[1] for r in riesgo]
            bars = ax.barh(nombres, vals, color="#f87171", edgecolor=GRID, height=0.5)
            ax.bar_label(bars, fmt="%.3f", color="#94a3b8", fontsize=9, padding=3)
            ax.set_xlim(0, 0.32); ax.set_title("Correlación con Attrition (+)", color="#f8fafc", fontsize=11)
            mpl_defaults(ax,fig); plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("#### 🟢 Top 5: RETIENEN al empleado")
            retencion = [("JobLevel",-0.257),("TotalWorkingYears",-0.244),("YearsAtCompany",-0.176),("MonthlyIncome",-0.170),("Age",-0.159)]
            fig,ax = plt.subplots(figsize=(5,3))
            nombres = [r[0] for r in retencion]; vals = [abs(r[1]) for r in retencion]
            bars = ax.barh(nombres, vals, color="#34d399", edgecolor=GRID, height=0.5)
            ax.bar_label(bars, fmt="%.3f", color="#94a3b8", fontsize=9, padding=3)
            ax.set_xlim(0, 0.32); ax.set_title("Correlación con Attrition (−)", color="#f8fafc", fontsize=11)
            mpl_defaults(ax,fig); plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════
# RESULTADOS POR MODELO
# ══════════════════════════════════════════════════════════════════
elif seccion == "🧪 Resultados por modelo":
    st.markdown("## 🧪 Resultados por modelo")
    modelo_sel = st.selectbox("Seleccionar modelo:", list(RESULTADOS.keys()))
    d = RESULTADOS[modelo_sel]
    color = COLORS[modelo_sel]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",  f"{d['accuracy']:.4f}")
    c2.metric("Precision", f"{d['precision']:.4f}")
    c3.metric("Recall",    f"{d['recall']:.4f}")
    c4.metric("F1-Score",  f"{d['f1']:.4f}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Matriz de Confusión")
        tp,fp,fn,tn = d["tp"],d["fp"],d["fn"],d["tn"]
        cm = np.array([[tn,fp],[fn,tp]], dtype=float)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        fig, ax = plt.subplots(figsize=(4.5,3.5))
        cmap = LinearSegmentedColormap.from_list("custom", [CARD, "#1e3a5f", color])
        im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred: No (0)","Pred: Yes (1)"], color="#94a3b8", fontsize=9)
        ax.set_yticklabels(["Real: No (0)","Real: Yes (1)"], color="#94a3b8", fontsize=9)
        labels = [["TN","FP"],["FN","TP"]]
        raw = [[tn,fp],[fn,tp]]
        for i in range(2):
            for j in range(2):
                ax.text(j,i, f"{labels[i][j]}={int(raw[i][j])}\n({cm_pct[i,j]:.1f}%)",
                        ha='center', va='center', color="white", fontsize=11, fontweight='bold')
        ax.set_title(f"Matriz de Confusión · {modelo_sel}", color="#f8fafc", fontsize=11, pad=10)
        fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### Perfil de métricas")
        metricas = ["Accuracy","Precision","Recall","F1-Score"]
        vals = [d["accuracy"],d["precision"],d["recall"],d["f1"]]
        fig, ax = plt.subplots(figsize=(4.5,3.5))
        bars = ax.barh(metricas, vals, color=color, edgecolor=GRID, height=0.5)
        ax.bar_label(bars, fmt="%.4f", color="#94a3b8", fontsize=10, padding=4)
        ax.set_xlim(0, 1.15)
        ax.axvline(x=0.5, color=GRID, linewidth=1, linestyle="--")
        ax.set_title(f"Métricas · {modelo_sel}", color="#f8fafc", fontsize=11)
        mpl_defaults(ax,fig); plt.tight_layout(); st.pyplot(fig); plt.close()

        c1,c2 = st.columns(2)
        c1.metric("✅ TP", tp, f"{tp/(tp+fn)*100:.0f}% renuncias detectadas")
        c2.metric("❌ FN", fn, f"renuncias perdidas")
        c1.metric("✅ TN", tn, f"{tn/(tn+fp)*100:.0f}% retenciones OK")
        c2.metric("❌ FP", fp, "falsas alarmas")

# ══════════════════════════════════════════════════════════════════
# COMPARATIVA
# ══════════════════════════════════════════════════════════════════
elif seccion == "📈 Comparativa":
    st.markdown("## 📈 Comparativa de todos los modelos")

    modelos = list(RESULTADOS.keys())
    df_comp = pd.DataFrame([{"Modelo":m,"Accuracy":RESULTADOS[m]["accuracy"],
        "Precision":RESULTADOS[m]["precision"],"Recall":RESULTADOS[m]["recall"],
        "F1-Score":RESULTADOS[m]["f1"]} for m in modelos]).sort_values("F1-Score",ascending=False).reset_index(drop=True)
    df_comp.index += 1
    st.dataframe(df_comp.style
        .format({"Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}","F1-Score":"{:.4f}"})
        .highlight_max(axis=0, color="#14432a")
        .highlight_min(axis=0, color="#3d1010"),
        use_container_width=True)
    st.markdown("---")

    metrica = st.selectbox("Métrica para comparar:", ["Accuracy","Precision","Recall","F1-Score"])
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### {metrica} por modelo")
        df_s = df_comp.sort_values(metrica, ascending=True)
        fig, ax = plt.subplots(figsize=(5,4))
        bars = ax.barh(df_s["Modelo"], df_s[metrica],
                       color=[COLORS[m] for m in df_s["Modelo"]], edgecolor=GRID, height=0.55)
        ax.bar_label(bars, fmt="%.4f", color="#94a3b8", fontsize=9, padding=3)
        ax.set_xlim(0, 1.15); ax.set_title(metrica, color="#f8fafc", fontsize=11)
        mpl_defaults(ax,fig); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### Precision vs Recall")
        fig, ax = plt.subplots(figsize=(5,4))
        # Curvas iso-F1
        r_range = np.linspace(0.05,1,200)
        for f1v in [0.2,0.3,0.4,0.5]:
            p_iso = f1v*r_range/(2*r_range-f1v)
            mask = (p_iso >= 0) & (p_iso <= 1)
            ax.plot(r_range[mask], p_iso[mask], color=GRID, linewidth=0.8, linestyle="--")
            idx = np.argmin(np.abs(r_range - 0.7))
            if mask[idx]:
                ax.text(r_range[idx], p_iso[idx]+0.02, f"F1={f1v}", color="#334155", fontsize=7)
        for m in modelos:
            d = RESULTADOS[m]
            ax.scatter(d["recall"], d["precision"], color=COLORS[m], s=90, zorder=5, edgecolors="white", linewidths=0.5)
            ax.annotate(m.split()[0], (d["recall"], d["precision"]), textcoords="offset points",
                        xytext=(5,4), fontsize=7.5, color="#94a3b8")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title("Precision vs Recall (iso-F1 punteado)", color="#f8fafc", fontsize=11)
        mpl_defaults(ax,fig); plt.tight_layout(); st.pyplot(fig); plt.close()

    # Heatmap
    st.markdown("#### Heatmap de métricas")
    met_keys = ["accuracy","precision","recall","f1"]
    met_labels = ["Accuracy","Precision","Recall","F1-Score"]
    z = np.array([[RESULTADOS[m][k] for k in met_keys] for m in modelos])
    fig, ax = plt.subplots(figsize=(8,4))
    cmap = LinearSegmentedColormap.from_list("hm",["#1e293b","#1e3a5f","#7dd3fc"])
    im = ax.imshow(z, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(met_labels, color="#94a3b8", fontsize=10)
    ax.set_yticks(range(len(modelos))); ax.set_yticklabels(modelos, color="#94a3b8", fontsize=9)
    for i in range(len(modelos)):
        for j in range(4):
            ax.text(j,i,f"{z[i,j]:.3f}", ha='center', va='center', color="white", fontsize=9, fontweight='bold')
    ax.set_title("Comparativa completa de métricas", color="#f8fafc", fontsize=12, pad=10)
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════
# ANÁLISIS CRÍTICO
# ══════════════════════════════════════════════════════════════════
elif seccion == "🔬 Análisis Crítico":
    st.markdown("## 🔬 Análisis Crítico & Conclusiones")

    col1, col2 = st.columns([1.1,1])
    with col1:
        st.markdown("### 📌 Hallazgos clave")
        hallazgos = [
            ("🏆","#34d399","Random Forest — mejor accuracy (87.41%) y precision (71.43%)",
             "Evita falsos positivos. Sin embargo, su recall bajo (21.74%) significa que pierde ~36 renuncias reales. Recomendado como modelo principal con ajuste de umbral."),
            ("📊","#fb923c","Naive Bayes — mejor recall (60.87%) y F1-Score (43.75%)",
             "Detecta más renuncias reales (28 de 46), pero genera 54 falsas alarmas. Útil cuando el costo de perder una renuncia es mayor al de intervenir innecesariamente."),
            ("⚠️","#f87171","Desbalance de clases (84% / 16%)",
             "El principal obstáculo. Infla el accuracy y penaliza el recall. Se recomienda SMOTE o ajuste del umbral de decisión para la clase positiva."),
            ("🔬","#2dd4bf","Gamma PYDRA — funcional pero costoso O(n²)",
             "Correcto y tolerante a valores perdidos, pero inviable para datasets grandes. Muestra que el diseño teórico no siempre escala a producción."),
            ("💡","#fbbf24","Data Leakage identificado y reconocido",
             "Imputer y Scaler se ajustaron antes del split. Las métricas reportadas pueden ser ligeramente optimistas. El equipo lo documentó explícitamente."),
        ]
        for icon, color, titulo, texto in hallazgos:
            st.markdown(f"""<div style='background:#111827;border:1px solid #1e293b;border-left:3px solid {color};border-radius:10px;padding:14px;margin-bottom:8px;'>
                <b style='color:#e2e8f0;'>{icon} {titulo}</b><br>
                <span style='color:#94a3b8;font-size:13px;line-height:1.6;'>{texto}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 Recomendación final")
        st.markdown("""<div style='background:#111827;border:1px solid #1e293b;border-left:3px solid #f87171;border-radius:10px;padding:20px;margin-bottom:16px;'>
            <p style='color:#7dd3fc;font-family:Space Mono;font-size:11px;margin-bottom:8px;letter-spacing:2px;'>MODELO RECOMENDADO</p>
            <h3 style='color:#f8fafc;font-size:1.4rem;margin:0 0 8px;'>🌲 Random Forest</h3>
            <p style='color:#94a3b8;font-size:13px;line-height:1.7;margin:0;'>
            Mejor desempeño general con alta precision. Se recomienda ajustar el 
            umbral de decisión (p.ej. 0.30 en lugar de 0.50) para mejorar el recall 
            sin sacrificar demasiada precisión en el contexto de RRHH.
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("### 🚀 Mejoras propuestas")
        mejoras = [
            "Aplicar SMOTE para balancear clases en entrenamiento",
            "Ajustar umbral de decisión (0.3 en lugar de 0.5)",
            "Corregir data leakage (fit solo en X_train)",
            "Cross-validation estratificada (k-fold = 5 o 10)",
            "Búsqueda de hiperparámetros con GridSearchCV",
            "Agregar métricas AUC-ROC y curva Precision-Recall",
        ]
        for i, m in enumerate(mejoras, 1):
            st.markdown(f"<div style='display:flex;gap:10px;background:#111827;border:1px solid #1e293b;border-radius:8px;padding:10px 14px;margin-bottom:4px;'><span style='font-family:Space Mono;color:#7dd3fc;font-size:11px;min-width:24px;'>{i:02d}</span><span style='color:#cbd5e1;font-size:13px;'>{m}</span></div>", unsafe_allow_html=True)

        st.markdown("### RF vs Naive Bayes: trade-off clave")
        fig, ax = plt.subplots(figsize=(5,2.8))
        x = np.arange(4); w = 0.35
        met_labels = ["Accuracy","Precision","Recall","F1-Score"]
        rf_vals  = [0.8741,0.7143,0.2174,0.3333]
        nb_vals  = [0.7891,0.3415,0.6087,0.4375]
        bars1 = ax.bar(x-w/2, rf_vals, w, label="Random Forest", color="#f87171", edgecolor=GRID)
        bars2 = ax.bar(x+w/2, nb_vals, w, label="Naive Bayes",   color="#fb923c", edgecolor=GRID)
        ax.set_xticks(x); ax.set_xticklabels(met_labels, fontsize=9)
        ax.set_ylim(0, 1.1); ax.legend(fontsize=8, facecolor=CARD, labelcolor="#94a3b8")
        ax.bar_label(bars1, fmt="%.2f", color="#94a3b8", fontsize=7.5, padding=2)
        ax.bar_label(bars2, fmt="%.2f", color="#94a3b8", fontsize=7.5, padding=2)
        mpl_defaults(ax,fig); plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("""<div style='background:#111827;border:1px solid #1e293b;border-left:3px solid #7dd3fc;border-radius:10px;padding:20px;'>
        <b style='color:#7dd3fc;font-family:Space Mono;font-size:11px;letter-spacing:2px;'>CONCLUSIÓN DEL EQUIPO 12</b>
        <p style='color:#cbd5e1;font-size:14px;line-height:1.8;margin-top:10px;margin-bottom:0;'>
        De los 8 clasificadores evaluados, <b>Random Forest</b> obtuvo el mejor desempeño general.
        <b>SVM</b> y <b>Regresión Logística</b> son alternativas sólidas e interpretables.
        <b>Naive Bayes</b> destaca en recall: útil si la prioridad es no perder renuncias.
        El clasificador <b>Gamma PYDRA</b> es funcionalmente correcto pero computacionalmente inviable a escala.
        <b>KNN</b> y <b>Perceptrón</b> resultaron los modelos más débiles. El principal reto
        del problema es el <b>desbalance de clases</b>, que limita el recall de todos los modelos
        y debe abordarse en trabajos futuros.
        </p>
    </div>""", unsafe_allow_html=True)
