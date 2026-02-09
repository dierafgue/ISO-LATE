# =============================================================================
# 📘 APP STREAMLIT – ISO-LATE - DESARROLLADO POR DIEGO GUERRERO MDI.
# =============================================================================
import streamlit as st
import numpy as np
from funciones_usuario import *  # (ideal: luego lo cambiamos por imports puntuales)
# from PIL import Image  # ❌ No se usa aquí → elimínalo

# =============================================================================
# === CONFIGURACIÓN INICIAL ==================================================
# =============================================================================
st.set_page_config(
    page_title="ISO-LATE",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"  # ✅ ya que lo ocultas, mejor colapsado
)

# =============================================================================
# === CSS (solo una vez) =====================================================
# =============================================================================
@st.cache_data(show_spinner=False)
def _get_css() -> str:
    return """
    <style>
    /* =========================
       ESTILO GLOBAL
    ========================= */
    body {
        background-color: #f6f8fb;
        color: #222831;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }

    /* =========================
       OCULTAR SIDEBAR COMPLETO
    ========================= */
    [data-testid="stSidebar"] { display: none; }

    section[data-testid="stMain"] {
        margin-left: 0 !important;
        width: 100% !important;
    }

    /* =========================
       BOTONES
    ========================= */
    .stButton > button {
        border-radius: 8px;
        background-color: #30475e;
        color: white;
        border: none;
        font-weight: 500;
        padding: 0.5rem 1.2rem;
    }
    .stButton > button:hover { background-color: #3e5a78; }

    /* =========================
       INPUTS
    ========================= */
    input, textarea, select { border-radius: 6px !important; }
    </style>
    """

st.markdown(_get_css(), unsafe_allow_html=True)

# =============================================================================
# === SECCIÓN 1: PARÁMETROS GENERALES DEL MODELO (DISEÑO SIMÉTRICO) ==========
# =============================================================================
st.title("💻 ISO-LATE")
st.header("📋 Parámetros generales del modelo estructural")
st.markdown("Define geometría, secciones y material del modelo base.")

col_geo, col_sec, col_mat = st.columns(3, gap="large")

# -------------------------------------------------------------------------
# ⚙️ GEOMETRÍA
# -------------------------------------------------------------------------
with col_geo:
    st.subheader("⚙️ Geometría")
    c1, c2 = st.columns(2)
    n_pisos = c1.number_input("N° de pisos", min_value=1, max_value=30, value=2, step=1)
    n_vanos = c2.number_input("N° de vanos", min_value=1, max_value=8, value=1, step=1)

    st.caption("Rangos permitidos: pisos 1–30 | vanos 1–8")

    l_vano = st.number_input("Longitud de vano [m]", min_value=1.0, value=5.0, step=0.5)
    h_piso_1 = round(st.number_input("Altura 1er piso [m]", min_value=2.0, value=4.0, step=0.1), 6)
    h_piso_restantes = round(st.number_input("Altura pisos restantes [m]", min_value=2.0, value=3.0, step=0.1), 6)

# -------------------------------------------------------------------------
# 📏 SECCIONES (Básico / Avanzado)
# -------------------------------------------------------------------------
with col_sec:
    st.subheader("📏 Propiedades de secciones")
    modo_avanzado = st.checkbox("🔧 Modo avanzado", value=False)

    if not modo_avanzado:
        st.caption("Modo básico (dimensiones en cm)")

        st.markdown("#### 🧱 Columna")
        cc1, cc2 = st.columns(2)
        b_col = cc1.number_input("Base columna [cm]", value=50.0, step=0.5)
        h_col = cc2.number_input("Altura columna [cm]", value=50.0, step=0.5)

        st.markdown("#### 🪵 Viga")
        cv1, cv2 = st.columns(2)
        b_viga = cv1.number_input("Base viga [cm]", value=30.0, step=0.5)
        h_viga = cv2.number_input("Altura viga [cm]", value=50.0, step=0.5)

        # cm → m
        bcol_m, hcol_m = b_col / 100.0, h_col / 100.0
        bvig_m, hvig_m = b_viga / 100.0, h_viga / 100.0

        A_col = bcol_m * hcol_m
        I_col = bcol_m * (hcol_m**3) / 12.0
        A_viga = bvig_m * hvig_m
        I_viga = bvig_m * (hvig_m**3) / 12.0

    else:
        st.caption("Modo avanzado (propiedades en cm² / cm⁴)")

        st.markdown("#### 🧱 Columna")
        ca1, ca2 = st.columns(2)
        A_col_cm2 = ca1.number_input("Área columna [cm²]", value=2500.00, step=10.0)
        I_col_cm4 = ca2.number_input("Inercia columna [cm⁴]", value=520833.33, step=100.0)

        st.markdown("#### 🪵 Viga")
        cb1, cb2 = st.columns(2)
        A_viga_cm2 = cb1.number_input("Área viga [cm²]", value=1500.00, step=10.0)
        I_viga_cm4 = cb2.number_input("Inercia viga [cm⁴]", value=312500.00, step=100.0)

        # cm² → m² ; cm⁴ → m⁴
        A_col = A_col_cm2 / 1e4
        I_col = I_col_cm4 / 1e8
        A_viga = A_viga_cm2 / 1e4
        I_viga = I_viga_cm4 / 1e8

# -------------------------------------------------------------------------
# 🧱 MATERIAL Y CARGAS
# -------------------------------------------------------------------------
with col_mat:
    st.subheader("🧱 Material y cargas")

    cm1, cm2 = st.columns(2)
    E = cm1.number_input("Módulo E [Tf/m²]", value=2534563.54, step=10000.0)
    peso_especifico = cm2.number_input("Peso específico [Tf/m³]", value=2.4028, step=0.1)

    st.markdown("#### ⚖️ Cargas")
    sobrecarga_muerta = st.number_input("Sobrecarga muerta [Tf/m]", value=0.0, step=1.0)
    amortiguamiento = st.number_input("Amortiguamiento ζ (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)

# -------------------------------------------------------------------------
# ✅ Almacenamiento (solo si cambia algo)
# -------------------------------------------------------------------------
params_nuevos = {
    "n_pisos": n_pisos,
    "n_vanos": n_vanos,
    "l_vano": l_vano,
    "h_piso_1": h_piso_1,
    "h_piso_restantes": h_piso_restantes,
    "E": E,
    "I_col": I_col,
    "A_col": A_col,
    "I_viga": I_viga,
    "A_viga": A_viga,
    "peso_especifico": peso_especifico,
    "sobrecarga_muerta": sobrecarga_muerta,
    "amortiguamiento": amortiguamiento / 100.0,
}

# ✅ Evita reescritura constante del session_state en cada rerun
if st.session_state.get("param_estruct") != params_nuevos:
    st.session_state["param_estruct"] = params_nuevos
    # compatibilidad con el resto de tu app (si lo usa)
    st.session_state["A_col"] = A_col
    st.session_state["I_col"] = I_col
    st.session_state["A_viga"] = A_viga
    st.session_state["I_viga"] = I_viga

st.markdown("---")

# =============================================================================
# === SECCIÓN 2: MODELO + MASAS + TRANSFORMACIÓN + CONDENSACIÓN + PLOT ========
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

st.markdown("## 🏗️ Definición geométrica y estructural del modelo")
st.markdown(
    "Genera la geometría del pórtico, ensambla la rigidez global y calcula masas/matrices asociadas. "
    "Se asume **diafragma rígido por piso**: el sistema se condensa a **1 GDL horizontal (Ux) por planta**. "
    "En cada nodo se consideran **2 GDL locales**: **VY** y **giro (θ)**."
)

# --- Layout principal ---
col_left, col_right = st.columns([0.50, 0.50], gap="large")

# =========================
# IZQUIERDA: Control
# =========================
with col_left:
    st.markdown("### ⚙️ Control del modelo")

    # ✅ Solo habilitar si existen parámetros
    hay_params = "param_estruct" in st.session_state and st.session_state["param_estruct"] is not None
    generar = st.button(
        "🚀 Generar modelo estructural",
        use_container_width=True,
        disabled=not hay_params
    )
    if not hay_params:
        st.info("Primero define los parámetros en la **Sección 1**.")

# -----------------------------------------------------------------------------
# Helpers: recuperar params + hash sencillo para evitar recomputar innecesario
# -----------------------------------------------------------------------------
def _get_params() -> dict:
    params = st.session_state.get("param_estruct", {}) or {}
    # Defaults seguros (los mismos que tú usabas)
    return {
        "n_pisos": int(params.get("n_pisos", 2)),
        "n_vanos": int(params.get("n_vanos", 1)),
        "l_vano": float(params.get("l_vano", 5.0)),
        "h_piso_1": round(float(params.get("h_piso_1", 4.0)), 6),
        "h_piso_restantes": round(float(params.get("h_piso_restantes", 3.0)), 6),
        "E": float(params.get("E", 21316773.9449)),
        "I_col": float(params.get("I_col", 0.0052)),
        "A_col": float(params.get("A_col", 0.25)),
        "I_viga": float(params.get("I_viga", 0.0031)),
        "A_viga": float(params.get("A_viga", 0.15)),
        "peso_especifico": float(params.get("peso_especifico", 2.4)),
        "sobrecarga_muerta": float(params.get("sobrecarga_muerta", 38.0)),
        "amortiguamiento": float(params.get("amortiguamiento", 0.05)),
    }

def _params_key(p: dict) -> tuple:
    # clave estable para detectar cambios de input
    return (
        p["n_pisos"], p["n_vanos"], p["l_vano"], p["h_piso_1"], p["h_piso_restantes"],
        p["E"], p["I_col"], p["A_col"], p["I_viga"], p["A_viga"],
        p["peso_especifico"], p["sobrecarga_muerta"], p["amortiguamiento"]
    )

# -----------------------------------------------------------------------------
# Generación: solo si se presiona el botón o si cambian parámetros y tú quieres autorefresh
# -----------------------------------------------------------------------------
if generar:
    try:
        p = _get_params()
        pkey = _params_key(p)

        # ✅ sincronizar (si tu app lo usa luego)
        st.session_state["peso_especifico"]   = p["peso_especifico"]
        st.session_state["sobrecarga_muerta"] = p["sobrecarga_muerta"]
        st.session_state["amortiguamiento"]   = p["amortiguamiento"]
        st.session_state["E"]                 = p["E"]
        st.session_state["A_col"]             = p["A_col"]
        st.session_state["I_col"]             = p["I_col"]
        st.session_state["A_viga"]            = p["A_viga"]
        st.session_state["I_viga"]            = p["I_viga"]

        n_pisos, n_vanos = p["n_pisos"], p["n_vanos"]
        l_vano = p["l_vano"]
        h_piso_1, h_piso_restantes = p["h_piso_1"], p["h_piso_restantes"]
        E, I_col, A_col = p["E"], p["I_col"], p["A_col"]
        I_viga, A_viga = p["I_viga"], p["A_viga"]
        peso_especifico, sobrecarga_muerta = p["peso_especifico"], p["sobrecarga_muerta"]

        # =============================
        # 1) Nodos
        # =============================
        nodes = []
        y_actual = 0.0
        for i in range(n_pisos + 1):
            base_id = i * (n_vanos + 1)
            for j in range(n_vanos + 1):
                nodes.append((j * l_vano, y_actual, base_id + j))
            y_actual += h_piso_1 if i == 0 else h_piso_restantes

        # =============================
        # 2) GDL map
        # =============================
        gdl_map = generar_gdl_map(nodes)

        # =============================
        # 3) Propiedades
        # =============================
        propiedades = {
            "col":  {"E": E, "I": I_col,  "A": A_col},
            "viga": {"E": E, "I": I_viga, "A": A_viga},
        }

        # =============================
        # 4) Conectividades
        # =============================
        element_node_pairs = []

        # Columnas
        for i in range(n_pisos):
            row_i = i * (n_vanos + 1)
            row_j = (i + 1) * (n_vanos + 1)
            for j in range(n_vanos + 1):
                element_node_pairs.append((row_i + j, row_j + j, "col"))

        # Vigas (de piso 1 a n_pisos)
        for i in range(1, n_pisos + 1):
            row_i = i * (n_vanos + 1)
            for j in range(n_vanos):
                element_node_pairs.append((row_i + j, row_i + j + 1, "viga"))

        # =============================
        # 5) Ensamble global
        # =============================
        elements = []
        for n1, n2, tipo in element_node_pairs:
            node_i = nodes[n1]
            node_j = nodes[n2]
            prop = propiedades[tipo]
            elements.append(Element(node_i, node_j, prop["E"], prop["I"], prop["A"], gdl_map))

        total_dofs = max(dof for el in elements for dof in el.dofs if dof is not None) + 1
        K_global = assemble_global_stiffness(elements, total_dofs)

        # =============================
        # 6) Masa condensada por piso
        # =============================
        if "calcular_matriz_masas_por_piso" not in globals():
            st.error("No se encontró `calcular_matriz_masas_por_piso` en tus imports.")
            M_cond = None
        else:
            M_cond = calcular_matriz_masas_por_piso(
                nodes,
                element_node_pairs,
                propiedades,
                peso_especifico=peso_especifico,
                sobrecarga_muerta=sobrecarga_muerta,
            )

        # =============================
        # 7) Matriz T (solo si hay M_cond)
        # =============================
        T = None
        if M_cond is not None:
            nodes_arr = np.array(nodes, dtype=float)
            alturas = np.unique(nodes_arr[:, 1])
            pisos_y_emp = [round(float(y), 6) for y in alturas if float(y) > 0.0]
            altura_a_col = {y: i for i, y in enumerate(pisos_y_emp)}

            T = np.zeros((K_global.shape[0], len(pisos_y_emp)))
            for (x, y, nid) in nodes:
                if y > 0:
                    dof_vx = gdl_map.get((nid, "vx"))
                    if dof_vx is not None:
                        T[int(dof_vx), altura_a_col[round(float(y), 6)]] = 1.0

        # =============================
        # 8) Diafragma rígido + condensación a 1 GDL/piso + chequeo
        # =============================
        K_cond = None
        k_modelo = None
        k_aprox = None
        ratio_k = None

        nodes_arr = np.array(nodes, dtype=float)
        alturas = np.unique(nodes_arr[:, 1])
        pisos_y = [round(float(y), 6) for y in alturas if float(y) > 0.0]
        n_pisos_cond = len(pisos_y)

        if n_pisos_cond == 0:
            st.error("No hay pisos (y>0). Revisa nodes/n_pisos.")
        else:
            # (A) Master por piso: menor x
            master_node_por_piso = {}
            for y in pisos_y:
                nodos_en_y = [(float(x), int(nid)) for (x, yy, nid) in nodes if round(float(yy), 6) == y]
                nodos_en_y.sort(key=lambda t: t[0])
                master_node_por_piso[y] = nodos_en_y[0][1]

            # DOF Ux de masters
            dofUx_piso = []
            for y in pisos_y:
                nid_m = master_node_por_piso[y]
                dof_m = gdl_map.get((nid_m, "vx"))
                if dof_m is None:
                    raise ValueError(f"Master sin vx en y={y}. Revisa gdl_map.")
                dofUx_piso.append(int(dof_m))

            # DOFs completos
            vx_all = [int(dof) for (nid, tipo), dof in gdl_map.items() if tipo == "vx" and dof is not None]
            vy_all = [int(dof) for (nid, tipo), dof in gdl_map.items() if tipo == "vy" and dof is not None]
            th_all = [int(dof) for (nid, tipo), dof in gdl_map.items() if tipo == "theta" and dof is not None]

            set_masters = set(dofUx_piso)
            vx_slaves = sorted([d for d in vx_all if d not in set_masters])
            gdl_ss = sorted(vx_slaves + vy_all + th_all)

            n_full = K_global.shape[0]
            n_p = n_pisos_cond
            n_s = len(gdl_ss)

            # Map interno -> col
            s_col = {dof: j for j, dof in enumerate(gdl_ss)}

            # Map vx -> piso (SIN usar pisos_y.index en loop)
            piso_index = {y: i for i, y in enumerate(pisos_y)}
            dof_to_piso = {}
            for (x, yy, nid) in nodes:
                yk = round(float(yy), 6)
                if yk in piso_index:
                    dof_vx = gdl_map.get((nid, "vx"))
                    if dof_vx is not None:
                        dof_to_piso[int(dof_vx)] = piso_index[yk]

            # R
            R = np.zeros((n_full, n_p + n_s), dtype=float)

            # 1) vx: todos al Ux del piso
            for dof_vx in vx_all:
                piso_idx = dof_to_piso.get(int(dof_vx))
                if piso_idx is not None:
                    R[int(dof_vx), piso_idx] = 1.0

            # 2) internos: identidad
            for dof in gdl_ss:
                R[int(dof), n_p + s_col[int(dof)]] = 1.0

            Kd = R.T @ K_global @ R
            Kd = 0.5 * (Kd + Kd.T)

            # Condensación estática
            Kpp = Kd[:n_p, :n_p]
            if n_s == 0:
                K_cond = Kpp.copy()
            else:
                Kss = Kd[n_p:, n_p:]
                Kps = Kd[:n_p, n_p:]
                Ksp = Kd[n_p:, :n_p]
                try:
                    Kss_inv = inv(Kss)
                except np.linalg.LinAlgError:
                    Kss_inv = np.linalg.pinv(Kss)
                    st.warning("Kss singular; se usó pseudo-inversa (pinv).")
                K_cond = Kpp - Kps @ Kss_inv @ Ksp

            K_cond = 0.5 * (K_cond + K_cond.T)
            K_cond = np.array(K_cond, dtype=float, copy=True)

            # Chequeo k
            hs = np.array([h_piso_1] + [h_piso_restantes] * (n_pisos_cond - 1), dtype=float)
            n_col = n_vanos + 1
            k_aprox = n_col * (12.0 * E * I_col) / (hs ** 3)

            k_modelo = np.zeros(n_pisos_cond, dtype=float)
            if n_pisos_cond == 1:
                k_modelo[0] = K_cond[0, 0]
            else:
                for i in range(1, n_pisos_cond):
                    k_modelo[i] = -K_cond[i - 1, i]
                k_modelo[0] = K_cond[0, 0] + K_cond[0, 1]

            ratio_k = k_modelo / k_aprox

        # =============================
        # Guardar en session_state (solo lo necesario)
        # =============================
        st.session_state["model_key"]          = pkey
        st.session_state["nodes"]              = nodes
        st.session_state["element_node_pairs"] = element_node_pairs
        st.session_state["propiedades"]        = propiedades
        st.session_state["gdl_map"]            = gdl_map
        st.session_state["elements"]           = elements
        st.session_state["K_global"]           = K_global
        st.session_state["M_cond"]             = M_cond
        st.session_state["T"]                  = T
        st.session_state["K_cond"]             = K_cond
        st.session_state["k_modelo"]           = k_modelo
        st.session_state["k_aprox"]            = k_aprox
        st.session_state["ratio_k"]            = ratio_k

        with col_left:
            st.success("✅ Modelo estructural generado correctamente.")
            st.markdown("#### 📊 Resumen del modelo")
            r1, r2, r3 = st.columns(3)
            r1.metric("Nodos", len(nodes))
            r2.metric("Elementos", len(elements))
            r3.metric("GDL", total_dofs)

    except Exception as e:
        st.error(f"⚠️ Error al generar el modelo: {e}")

# -----------------------------------------------------------------------------
# Mostrar resultados persistentes
# -----------------------------------------------------------------------------
requeridos = ["nodes", "element_node_pairs", "propiedades", "gdl_map", "elements", "K_global"]
faltantes = [k for k in requeridos if k not in st.session_state or st.session_state[k] is None]

with col_left:
    st.markdown("### 📌 Matrices")

    if faltantes:
        st.info("Primero genera el **modelo estructural**.")
    else:
        nodes              = st.session_state["nodes"]
        element_node_pairs = st.session_state["element_node_pairs"]
        propiedades        = st.session_state["propiedades"]
        gdl_map            = st.session_state["gdl_map"]
        elements           = st.session_state["elements"]
        K_global           = st.session_state["K_global"]

        M_cond = st.session_state.get("M_cond")
        T      = st.session_state.get("T")
        K_cond = st.session_state.get("K_cond")

        with st.expander("🧱 Ver matriz global de rigidez", expanded=False):
            st.write(f"Dimensión: {K_global.shape}")
            st.dataframe(np.round(K_global, 3), height=260, use_container_width=True)

        if M_cond is not None:
            with st.expander("⚖️ Ver matriz de masas condensada", expanded=False):
                st.write(f"Dimensión: {np.array(M_cond).shape}")
                st.dataframe(np.round(M_cond, 5), use_container_width=True)
        else:
            st.warning("M_cond no está listo. Revisa `calcular_matriz_masas_por_piso`.")

        if T is not None:
            with st.expander("🔁 Ver matriz de transformación", expanded=False):
                st.write(f"Dimensión: {T.shape}")
                st.dataframe(T.astype(int), use_container_width=True)

        if K_cond is not None:
            with st.expander("🧩 Ver matriz de rigidez condensada", expanded=False):
                st.write(f"Dimensión: {K_cond.shape}")
                st.dataframe(np.round(K_cond, 3), use_container_width=True)
        else:
            st.warning("K_cond no está listo. Revisa condensación / GDL.")

        k_modelo = st.session_state.get("k_modelo")
        k_aprox  = st.session_state.get("k_aprox")
        ratio_k  = st.session_state.get("ratio_k")

        if (k_modelo is not None) and (k_aprox is not None) and (ratio_k is not None):
            with st.expander("✅ Chequeo rápido", expanded=False):
                df_check = pd.DataFrame({
                    "Piso": np.arange(1, len(k_modelo) + 1),
                    "k_modelo (desde K_cond)": np.round(k_modelo, 6),
                    "k_aprox (Ncol*12EI/h^3)": np.round(k_aprox, 6),
                    "ratio (modelo/aprox)": np.round(ratio_k, 3),
                })
                st.dataframe(df_check, use_container_width=True)

# -----------------------------------------------------------------------------
# Derecha: gráfico
# -----------------------------------------------------------------------------
with col_right:
    st.markdown("### 🎨 Modelo estructural 2D")

    if faltantes:
        st.info("Aquí aparecerá el gráfico cuando generes el modelo.")
    else:
        # recuperar locals desde session_state (evita usar variables que podrían no existir)
        nodes    = st.session_state["nodes"]
        elements = st.session_state["elements"]
        gdl_map  = st.session_state["gdl_map"]
        propiedades = st.session_state["propiedades"]

        nodos_restringidos = [nid for (x, y, nid) in nodes if float(y) == 0.0]
        gdl_dinamicos_local = sorted(
            dof for (nid, tipo), dof in gdl_map.items()
            if dof is not None and tipo in ("vx", "vy", "theta")
        )

        fig = plot_structure(
            nodes=nodes,
            elements=elements,
            nodos_restringidos=nodos_restringidos,
            gdl_dinamicos_local=gdl_dinamicos_local,
            gdl_estaticos_local=[],
            gdl_map=gdl_map,
            propiedades=propiedades
        )

        if fig is not None and len(fig.axes) > 0:
            ax = fig.axes[0]
            ax.set_aspect(0.80)

            ys = [y for (_, y, _) in nodes]
            ymin, ymax = min(ys), max(ys)
            pad = 0.06 * (ymax - ymin) if ymax > ymin else 1.0
            ax.set_ylim(ymin - pad, ymax + pad)

            fig.set_size_inches(6.0, 3.8)
            fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)

            st.pyplot(fig, use_container_width=True)

        # CSS para quitar espacio extra (lo mantengo)
        st.markdown(
            """
            <style>
            div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div:has(> iframe[title="matplotlib.figure.Figure"]) + div {
                display: none !important;
            }
            iframe[title="matplotlib.figure.Figure"] { margin-bottom: -90px !important; }
            div[data-testid="stPlot"] { background: transparent !important; margin-bottom: -80px !important; }
            </style>
            """,
            unsafe_allow_html=True
        )

# =============================================================================
# ========== BLOQUE 3: NEC-24 (izq) + REGISTROS SÍSMICOS (der) ================
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import chardet

from numpy.linalg import inv, eig
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy import signal

st.markdown("## 📌 NEC-24 + Registro sísmico (antes del análisis modal)")

# ----------------------- Estilos (tus colores) -----------------------
BG         = "#2B3141"
COLOR_TEXT = "#E8EDF2"
COLOR_GRID = "#5B657A"

# =============================================================================
# Helpers (optimización): espectro NEC-24 vectorizado + caching
# =============================================================================
TABLA_FA = {
    "A": [0.90, 0.90, 0.90, 0.90, 0.90],
    "B": [1.00, 1.00, 1.00, 1.00, 1.00],
    "C": [1.40, 1.30, 1.23, 1.19, 1.13],
    "D": [1.60, 1.40, 1.25, 1.14, 1.00],
    "E": [1.80, 1.40, 1.10, 0.90, 0.62],
}
TABLA_FD = {
    "A": [0.90, 0.90, 0.90, 0.90, 0.90],
    "B": [1.00, 1.00, 1.00, 1.00, 1.00],
    "C": [1.36, 1.28, 1.15, 1.08, 1.00],
    "D": [1.62, 1.45, 1.28, 1.15, 1.00],
    "E": [2.10, 1.75, 1.65, 1.52, 1.36],
}
TABLA_FS = {
    "A": [0.75, 0.75, 0.75, 0.75, 0.75],
    "B": [0.75, 0.75, 0.75, 0.75, 0.75],
    "C": [0.85, 0.94, 1.06, 1.17, 1.28],
    "D": [1.02, 1.06, 1.19, 1.32, 1.44],
    "E": [1.50, 1.60, 1.80, 1.94, 2.09],
}
ZONAS_DICT = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4}

@st.cache_data(show_spinner=False)
def nec24_espectro(z: float, zona: str, suelo: str, R: float, T_final: float = 5.0, delta_t: float = 0.01):
    zona = zona.upper().strip()
    suelo = suelo.upper().strip()
    idx_zona = ZONAS_DICT[zona]

    Fa = float(TABLA_FA[suelo][idx_zona])
    Fd = float(TABLA_FD[suelo][idx_zona])
    Fs = float(TABLA_FS[suelo][idx_zona])

    r = 1.2
    T0 = 0.1 * Fs * Fd / Fa
    Tc = 0.45 * Fs * Fd / Fa
    TL = 2.4 * Fd

    T = np.linspace(0.0, T_final, int(T_final / delta_t) + 1)

    Sa_elast = np.zeros_like(T, dtype=float)

    m0 = (T == 0.0)
    m1 = (T > 0.0) & (T < T0)
    m2 = (T >= T0) & (T < Tc)
    m3 = (T >= Tc) & (T < TL)
    m4 = (T >= TL)

    Sa_elast[m0] = 0.0
    Sa_elast[m1] = z * Fa * (1.0 + 1.4 * (T[m1] / T0))
    Sa_elast[m2] = 2.4 * z * Fa
    Sa_elast[m3] = 2.4 * z * Fa * (Tc / T[m3]) ** r
    Sa_elast[m4] = 2.4 * z * Fa * (Tc / TL) ** r * (TL / T[m4]) ** 2

    Sa_inelas = np.zeros_like(T, dtype=float)
    mA = (T < Tc)
    mB = (T >= Tc) & (T < TL)
    mC = (T >= TL)

    Sa_inelas[mA] = (2.4 * z * Fa) / R
    Sa_inelas[mB] = (2.4 * z * Fa * (Tc / T[mB]) ** r) / R
    Sa_inelas[mC] = (2.4 * z * Fa * (Tc / TL) ** r * (TL / T[mC]) ** 2) / R

    # SDS y SD1 como tu criterio (T=1s)
    SDS = 2.4 * z * Fa
    T_SD1 = 1.0
    if T_SD1 < T0:
        SD1 = z * Fa * (1 + 1.4 * (T_SD1 / T0))
    elif T0 <= T_SD1 < Tc:
        SD1 = 2.4 * z * Fa
    elif Tc <= T_SD1 < TL:
        SD1 = 2.4 * z * Fa * (Tc / T_SD1) ** r
    else:
        SD1 = 2.4 * z * Fa * (Tc / TL) ** r * (TL / T_SD1) ** 2

    return T, Sa_elast, Sa_inelas, SDS, SD1, Fa, Fd, Fs, T_final, r, T0, Tc, TL


def _detectar_fuente(txt: str) -> str:
    t = (txt or "").lower()
    if ("pacific earthquake engineering research" in t) or ("peer strong motion" in t) or ("ngawest" in t) or (".at2" in t):
        return "PEER NGA"
    if ("red nacional de acelerógrafos" in t) or ("renac" in t) or ("igepn" in t):
        return "RENAC (IG-EPN)"
    if ("instituto geofísico" in t) and ("pucp" not in t):
        return "IGP"
    return "Desconocido"


# =============================================================================
# Layout principal
# =============================================================================
col_left, col_right = st.columns([1.05, 1.95], gap="large")

# =============================================================================
# IZQUIERDA: Parámetros NEC-24 + Espectro
# =============================================================================
with col_left:
    with st.container(border=True):
        st.subheader("🧩 Parámetros del espectro NEC-24")

        c1, c2 = st.columns(2)
        with c1:
            z = st.number_input("Intensidad sísmica (z)", 0.1, 1.0, 0.47, 0.01, key="nec_z")
            zona_sismica = st.selectbox("Zona sísmica", ["I", "II", "III", "IV", "V"], index=3, key="nec_zona")
            tipo_suelo   = st.selectbox("Tipo de suelo", ["A", "B", "C", "D", "E"], index=2, key="nec_suelo")
        with c2:
            R  = st.number_input("Factor de reducción (R)", 1.0, 10.0, 8.0, 0.1, key="nec_R")
            Ie = st.number_input("Factor de importancia (Ie)", 0.5, 2.0, 1.0, 0.1, key="nec_Ie")

        st.session_state["nec24_params"] = {
            "z": float(z),
            "zona": str(zona_sismica),
            "suelo": str(tipo_suelo),
            "R": float(R),
            "Ie": float(Ie),
        }

    with st.container(border=True):
        st.subheader("📈 Espectro de respuesta – NEC-24")

        T_spec, Sa_elast, Sa_inelas, SDS, SD1, Fa, Fd, Fs, T_final, *_ = nec24_espectro(
            z=float(z), zona=str(zona_sismica), suelo=str(tipo_suelo), R=float(R),
            T_final=5.0, delta_t=0.01
        )

        st.session_state["SDS"] = float(SDS)
        st.session_state["SD1"] = float(SD1)

        st.caption(f"**SDS = {SDS:.3f} g**, **SD1 = {SD1:.3f} g·s**")
        st.caption(f"Coeficientes: Fa={Fa:.2f}, Fd={Fd:.2f}, Fs={Fs:.2f}")

        fig, ax = plt.subplots(figsize=(6.6, 4.3))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.plot(T_spec, Sa_elast, lw=2.2, label="Elástico")
        ax.plot(T_spec, Sa_inelas, "--", lw=2.0, label=f"Inelástico (R={R:g})")
        ax.set_xlabel("Período T [s]", color=COLOR_TEXT)
        ax.set_ylabel("Sa [g]", color=COLOR_TEXT)
        ax.tick_params(colors=COLOR_TEXT)
        ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        leg = ax.legend(framealpha=0.95)
        for t in leg.get_texts():
            t.set_color("black")
        st.pyplot(fig, use_container_width=True)

# =============================================================================
# DERECHA: Registro sísmico
# =============================================================================
with col_right:
    with st.container(border=True):
        st.subheader("〰️ Carga del registro sísmico")
        col_ctrl, col_graf = st.columns([1.2, 2.5], gap="large")

        with col_ctrl:
            uploaded = st.file_uploader(
                "📁 Selecciona un archivo de aceleración (.txt o .at2)",
                type=["txt", "at2"]
            )
            aplicar_proc = st.checkbox("Aplicar filtrado + corrección de línea base", value=False)

        if uploaded is not None:
            # --- lectura binaria + encoding ---
            contenido_binario = uploaded.read()
            codificacion = chardet.detect(contenido_binario).get("encoding") or "utf-8"
            texto = contenido_binario.decode(codificacion, errors="ignore")

            fuente = _detectar_fuente(texto)

            # --- parseo (tu función) ---
            nombre, unidad, dt, ag = detectar_formato_y_extraer(texto)
            ag = np.asarray(ag, dtype=float).ravel()

            # --- a m/s² ---
            if unidad == "cm/s²":
                ag_orig = ag / 100.0
            elif unidad == "g":
                ag_orig = ag * 9.81
            elif unidad == "m/s²":
                ag_orig = ag
            else:
                st.error(f"Unidad no reconocida: {unidad}")
                st.stop()

            # --- tiempo ---
            dt = float(dt)
            t_ag = np.linspace(0.0, dt * (len(ag_orig) - 1), len(ag_orig))

            # --- integración cruda ---
            vel_orig = cumtrapz(ag_orig, t_ag, initial=0.0)
            desp_orig = cumtrapz(vel_orig, t_ag, initial=0.0)

            # --- procesado (si aplica) ---
            ag_proc = vel_proc = desp_proc = None
            if aplicar_proc:
                # baseline lineal en aceleración
                coef_a = np.polyfit(t_ag, ag_orig, 1)
                ag_bc = ag_orig - np.polyval(coef_a, t_ag)

                # butter bandpass fijo (0.10–25 Hz, orden 4), filtro causal
                fs = 1.0 / dt
                nyq = fs / 2.0
                low = 0.10 / nyq
                high = 25.0 / nyq
                b, a = signal.butter(4, [low, high], btype="band")
                ag_filt = signal.lfilter(b, a, ag_bc)

                vel_raw = cumtrapz(ag_filt, t_ag, initial=0.0)

                coef_v = np.polyfit(t_ag, vel_raw, 1)
                vel_proc = vel_raw - np.polyval(coef_v, t_ag)

                desp_raw = cumtrapz(vel_proc, t_ag, initial=0.0)

                coef_u = np.polyfit(t_ag, desp_raw, 2)
                desp_proc = desp_raw - np.polyval(coef_u, t_ag)

                ag_proc = ag_filt

            # --- guardar espectro en sesión (para diseño LRB) ---
            st.session_state["T_spec"] = np.array(T_spec, dtype=float).ravel()
            st.session_state["Sa_elast_spec"] = np.array(Sa_elast, dtype=float).ravel()
            st.session_state["Sa_inelas_spec"] = np.array(Sa_inelas, dtype=float).ravel()
            st.session_state["T_final_spec"] = float(T_final)
            st.session_state["R_spec"] = float(R)

            # --- info ---
            with col_ctrl:
                st.markdown(f"**Evento:** {nombre}")
                st.markdown(f"**Fuente detectada:** {fuente}")
                st.markdown(f"**Paso de tiempo:** {dt:.4f} s")
                st.markdown(f"**Duración total:** {t_ag[-1]:.2f} s")
                st.markdown(f"**Número de puntos:** {len(ag_orig)}")

            # --- estilos de curvas ---
            COLOR_ORIG = "#9DBEF7"
            COLOR_PROC = "#FFD479"
            LW_ORIG_SOLO = 0.5
            LW_ORIG_OVER = 0.25
            LW_PROC = 0.25

            # --- plot ---
            with col_graf:
                fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
                fig.patch.set_facecolor(BG)

                for ax in axs:
                    ax.set_facecolor(BG)
                    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
                    ax.tick_params(colors=COLOR_TEXT)
                    for s in ("top", "right"):
                        ax.spines[s].set_visible(False)

                axs[0].plot(t_ag, ag_orig, lw=(LW_ORIG_OVER if aplicar_proc else LW_ORIG_SOLO), color=COLOR_ORIG, label="Original")
                if aplicar_proc:
                    axs[0].plot(t_ag, ag_proc, lw=LW_PROC, color=COLOR_PROC, label="Filtrado + corregido")
                axs[0].set_ylabel("Aceleración [m/s²]", color=COLOR_TEXT)
                axs[0].set_title(f"Registro sísmico – {nombre}", color=COLOR_TEXT)

                axs[1].plot(t_ag, vel_orig, lw=(LW_ORIG_OVER if aplicar_proc else LW_ORIG_SOLO), color=COLOR_ORIG)
                if aplicar_proc:
                    axs[1].plot(t_ag, vel_proc, lw=LW_PROC, color=COLOR_PROC)
                axs[1].set_ylabel("Velocidad [m/s]", color=COLOR_TEXT)

                axs[2].plot(t_ag, desp_orig, lw=(LW_ORIG_OVER if aplicar_proc else LW_ORIG_SOLO), color=COLOR_ORIG)
                if aplicar_proc:
                    axs[2].plot(t_ag, desp_proc, lw=LW_PROC, color=COLOR_PROC)
                axs[2].set_ylabel("Desplazamiento [m]", color=COLOR_TEXT)
                axs[2].set_xlabel("Tiempo [s]", color=COLOR_TEXT)

                if aplicar_proc:
                    handles, labels = axs[0].get_legend_handles_labels()
                    legend = fig.legend(
                        handles, labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.012),
                        ncol=2,
                        frameon=True,
                        framealpha=0.8,
                        edgecolor=COLOR_GRID,
                        fontsize=10,
                        handlelength=2.8,
                        columnspacing=1.5,
                        borderpad=0.5
                    )
                    legend.get_frame().set_facecolor(BG)
                    legend.get_frame().set_edgecolor(COLOR_GRID)
                    for txt in legend.get_texts():
                        txt.set_color(COLOR_TEXT)

                fig.subplots_adjust(bottom=0.095, top=0.93)
                st.pyplot(fig)

            # --- guardar en sesión (MISMAS CLAVES) ---
            st.session_state["dt"] = float(dt)
            st.session_state["t_ag"] = t_ag
            st.session_state["ag_original"] = ag_orig
            st.session_state["vel_original"] = vel_orig
            st.session_state["desp_original"] = desp_orig

            if aplicar_proc:
                st.session_state["ag_filt"] = ag_proc
                st.session_state["vel"] = vel_proc
                st.session_state["desp"] = desp_proc
            else:
                st.session_state["ag_filt"] = ag_orig
                st.session_state["vel"] = vel_orig
                st.session_state["desp"] = desp_orig

            with col_ctrl:
                st.success("✅ Registro cargado correctamente.")

# =============================================================================
# === BLOQUE 4: DISEÑO DEL AISLADOR LRB (MODAL + RAYLEIGH + DISEÑO + GRÁFICO) ==
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
from numpy.linalg import inv, eig

st.markdown("## 🧱 Diseño del aislador LRB")

# -------------------------------------------------------------------------
# ✅ Validación previa
# -------------------------------------------------------------------------
if ("K_cond" not in st.session_state) or ("M_cond" not in st.session_state):
    st.info("⚙️ Primero genera el modelo estructural y la condensación.")
    st.stop()

Kc = st.session_state["K_cond"]
Mc = st.session_state["M_cond"]

# -------------------------------------------------------------------------
# ✅ Modal cacheado: solo recalcula si cambia K/M (clave simple)
# -------------------------------------------------------------------------
def _km_key(K, M) -> tuple:
    K = np.asarray(K); M = np.asarray(M)
    return (K.shape, M.shape, float(np.sum(K)), float(np.sum(M)))

km_key = _km_key(Kc, Mc)

if st.session_state.get("_km_key_modal") != km_key:
    A = inv(Mc) @ Kc
    w2, phi = eig(A)

    idx = np.argsort(w2.real)
    w2 = w2[idx].real
    phi = phi[:, idx].real

    phi_norm = np.zeros_like(phi)
    for i in range(phi.shape[1]):
        m_modal = phi[:, i].T @ Mc @ phi[:, i]
        phi_norm[:, i] = phi[:, i] / np.sqrt(m_modal)

    w = np.sqrt(np.maximum(w2, 0.0))     # rad/s
    with np.errstate(divide="ignore", invalid="ignore"):
        T_modos = np.where(w > 0, 2*np.pi / w, np.inf)  # s
        f_modos = np.where(T_modos != np.inf, 1 / T_modos, 0.0)  # Hz

    st.session_state["_km_key_modal"] = km_key
    st.session_state["w_sin"] = w
    st.session_state["T_sin"] = T_modos
    st.session_state["f_sin"] = f_modos

    # ✅ ambos nombres por compatibilidad con tus otros bloques
    st.session_state["phi_norm_sin"] = phi_norm
    st.session_state["v_norm_sin"]   = phi_norm

# Recuperar modal
w        = st.session_state["w_sin"]
T_modos  = st.session_state["T_sin"]
f_modos  = st.session_state["f_sin"]
phi_norm = st.session_state["phi_norm_sin"]

# -------------------------------------------------------------------------
# ✅ ÚNICO subtítulo debajo del título principal
# -------------------------------------------------------------------------
st.markdown("### ⚙️ Parámetros de diseño")

# -------------------------------------------------------------------------
# ✅ Layout: IZQ (parámetros + tabla modal compacta + diseño) / DER (gráfico)
# -------------------------------------------------------------------------
col_izq, col_der = st.columns([1.2, 1.8], gap="large")

with col_izq:
    # 1) Amortiguamiento objetivo
    ζ = st.number_input(
        "Amortiguamiento modal objetivo ζ",
        0.0, 0.2, 0.05, 0.005,
        key="zeta_ray"
    )
    # ✅ Guardar para que el bloque 6 no lo repita
    st.session_state["zeta_modal"] = float(ζ)

    # 2) Rayleigh (recalcula solo si cambia ζ o cambia el modelo)
    ray_key = (km_key, float(ζ))
    if st.session_state.get("_ray_key") != ray_key:
        w_use = np.asarray(w, dtype=float).ravel()

        if len(w_use) >= 2 and np.isfinite(w_use[0]) and np.isfinite(w_use[1]) and w_use[0] > 0 and w_use[1] > 0:
            ω1, ω2 = float(w_use[0]), float(w_use[1])
            A_mat = np.array([[1/(2*ω1), ω1/2],
                              [1/(2*ω2), ω2/2]], dtype=float)
            b_vec = np.array([ζ, ζ], dtype=float)
            α, β = np.linalg.solve(A_mat, b_vec)
            C = α * Mc + β * Kc
        else:
            α, β = 0.0, 0.0
            C = np.zeros_like(Kc)

        st.session_state["_ray_key"] = ray_key
        st.session_state["alpha_rayleigh"] = float(α)
        st.session_state["beta_rayleigh"]  = float(β)
        st.session_state["C_rayleigh"]     = C

    # 3) Datos modales (tabla comprimida)
    with st.expander("📌 Datos modales", expanded=False):
        tabla = np.vstack((np.arange(1, len(f_modos) + 1), f_modos, T_modos)).T
        df_modos = pd.DataFrame(tabla, columns=["Modo", "f [Hz]", "T [s]"]).round(5)

        st.dataframe(
            df_modos,
            hide_index=True,
            use_container_width=True,
            height=min(35 * (len(df_modos) + 1), 160)
        )

    st.markdown("---")

    # 4) Modo de diseño + inputs
    st.markdown("#### 🧰 Modo de diseño")

    modo = st.radio(
        "Modo de diseño",
        ["Automático", "Por período objetivo"],
        index=0,
        key="modo_lrb"
    )
    modo_automatico = (modo == "Automático")
    modo_periodo_objetivo = (modo == "Por período objetivo")

    T_objetivo = st.number_input(
        "T_objetivo [s]",
        0.5, 5.0, 2.5, 0.1,
        key="T_obj_lrb",
        disabled=modo_automatico
    )

    ejecutar = st.button("⚙️ Ejecutar diseño del aislador", key="btn_lrb")

    if ejecutar:
        faltantes = []
        for k in ["SD1", "SDS", "T_sin", "nodes"]:
            val = st.session_state.get(k)
            if val is None or (isinstance(val, (list, np.ndarray)) and len(val) == 0):
                faltantes.append(k)

        if faltantes:
            st.error(f"⚠️ Faltan datos previos: {', '.join(faltantes)}. Ejecuta el espectro NEC-24 antes del diseño.")
        else:
            try:
                resultados_ais = diseno_aislador_LRB(
                    SD1=st.session_state["SD1"],
                    SDS=st.session_state["SDS"],
                    T_sin=st.session_state["T_sin"],
                    Mc=Mc,
                    nodos_restringidos=[nid for (x, y, nid) in st.session_state["nodes"] if float(y) == 0.0],
                    modo_automatico=modo_automatico,
                    modo_periodo_objetivo=modo_periodo_objetivo,
                    T_objetivo=T_objetivo,
                )

                req = ["k_inicial_1ais", "k_post_1ais", "yield_1ais", "c_1ais", "keff_1ais", "D_M", "delta_L"]
                faltan = [k for k in req if k not in resultados_ais]

                if faltan:
                    st.error(f"❌ Faltan en resultados_ais: {faltan}")
                else:
                    # ✅ Guardar pack completo
                    st.session_state["res_aislador"] = resultados_ais

                    # ✅ Guardar llaves sueltas para que NO falle el BLOQUE 7
                    for k in req:
                        st.session_state[k] = float(resultados_ais[k])

                    st.markdown("""
                    <div style="
                        background-color:#1E2331;
                        color:#F4F6FA;
                        padding:15px;
                        border-radius:10px;
                        border:1px solid #3A4050;
                        font-family:Consolas, monospace;
                        margin-top:10px;">
                    <b>=== PROPIEDADES DEL AISLADOR LRB INDIVIDUAL ===</b><br>
                    🧱 Rigidez efectiva 𝐊ₑ𝑓𝑓 : {:.3f} Tonf/m<br>
                    💧 Amortiguamiento viscoso 𝐂ₑ𝑞 : {:.3f} Tonf·s/m<br>
                    📈 Rigidez inicial elástica 𝐊ₑ : {:.3f} Tonf/m<br>
                    📉 Rigidez postfluencia 𝐊ₚ : {:.3f} Tonf/m<br>
                    ⚙️ Fuerza de fluencia 𝐅ᵧ : {:.3f} Tonf<br>
                    📐 Relación de rigideces 𝐫 = 𝐊ₚ/𝐊ₑ : {:.3f}
                    </div>
                    """.format(
                        resultados_ais["keff_1ais"],
                        resultados_ais["c_1ais"],
                        resultados_ais["k_inicial_1ais"],
                        resultados_ais["k_post_1ais"],
                        resultados_ais["yield_1ais"],
                        resultados_ais["k_post_1ais"] / resultados_ais["k_inicial_1ais"],
                    ), unsafe_allow_html=True)

                    st.success("✅ Diseño del aislador completado correctamente.")

            except Exception as e:
                st.error(f"❌ Error durante el diseño del aislador: {e}")

with col_der:
    if "res_aislador" not in st.session_state:
        st.info("ℹ️ Ejecuta el diseño para ver el gráfico del aislador.")
    else:
        resultados_ais = st.session_state["res_aislador"]

        st.markdown("""
        ### 📊 Curva Característica Fuerza–Desplazamiento del Aislador LRB  
        **(Modelo Bilineal – Unidades: Tonf y m)**
        """)

        set_style_arctic_dark()

        fig, ax = plot_ciclo_histeretico_lrb(
            Ke=resultados_ais["k_inicial_1ais"],
            Kp=resultados_ais["k_post_1ais"],
            Fy=resultados_ais["yield_1ais"],
            dy=resultados_ais["delta_L"],
            D2=resultados_ais["D_M"],
            Keff_ref=resultados_ais["keff_1ais"],
        )
        st.pyplot(fig, use_container_width=True)

# =============================================================================
# === BLOQUE 5: MODAL + ESQUEMAS (FIJA vs AISLADA) OPTIMIZADO PARA HASTA 30 PISOS
# === FIXES PEDIDOS:
#   (1) MODOS: FIJA y AISLADA misma altura
#   (2) MODOS: máximo 6 subplots por fila (si <6, se ajusta sin blancos)
#   (3) ESQUEMAS: FIJA y AISLADA misma altura
#   (4) ESQUEMAS: evitar corrimientos de bolitas/textos (congelar límites)
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from numpy.linalg import inv, eig, solve

st.markdown("## 🌊 Análisis modal – Base fija vs Base aislada")

# -----------------------------------------------------------------
# ✅ PRERREQUISITOS
# -----------------------------------------------------------------
if "K_cond" not in st.session_state or "M_cond" not in st.session_state:
    st.info("⚙️ Primero genera el modelo FIJO (K_cond, M_cond).")
    st.stop()

if "nodes" not in st.session_state or "element_node_pairs" not in st.session_state or "propiedades" not in st.session_state:
    st.info("⚙️ Primero genera el modelo estructural (nodes, element_node_pairs, propiedades).")
    st.stop()

if "res_aislador" not in st.session_state:
    st.info("⚙️ Primero diseña el aislador (res_aislador).")
    st.stop()

K_fix = st.session_state["K_cond"]
M_fix = st.session_state["M_cond"]

nodes              = st.session_state["nodes"]
element_node_pairs = st.session_state["element_node_pairs"]
propiedades        = st.session_state["propiedades"]

res_ais   = st.session_state["res_aislador"]
keff_1ais = float(res_ais["keff_1ais"])

nodos_restringidos = [nid for (x, y, nid) in nodes if float(y) == 0.0]
n_aisladores       = len(nodos_restringidos)

# -----------------------------------------------------------------
# 🎨 Paleta Arctic Dark Pastel (igual estilo)
# -----------------------------------------------------------------
BG                = "#2B3141"
COLOR_TEXT        = "#E8EDF2"
COLOR_GRID        = "#5B657A"
COLOR_BASE        = "#F4A6A0"
COLOR_SPR         = "#7EB6FF"
COLOR_MASS        = "#FF9A9A"
COLOR_LABEL_SPR   = "#A5D8FF"
COLOR_LABEL_MASS  = "#FFDFA0"
HALO  = [pe.withStroke(linewidth=2.4, foreground=BG), pe.Normal()]

COLOR_MODO   = "#A8D5FF"
COLOR_INV    = "#F2A6A0"
COLOR_STRUCT = "#C3C9D8"
halo2 = [pe.withStroke(linewidth=2.0, foreground=BG), pe.Normal()]

# -----------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------
def modal_props(K, M):
    A = inv(M) @ K
    w2, V = eig(A)
    idx = np.argsort(w2.real)
    w2 = w2[idx].real
    V  = V[:, idx].real

    Vn = np.zeros_like(V)
    for i in range(V.shape[1]):
        mm = V[:, i].T @ M @ V[:, i]
        Vn[:, i] = V[:, i] / np.sqrt(mm)

    w = np.sqrt(np.maximum(w2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(w > 0, 2*np.pi/w, np.inf)
        f = np.where(T != np.inf, 1/T, 0.0)
    return w, T, f, Vn

def _freeze_axes_limits(fig):
    """Congela límites para evitar corrimientos al renderizar en Streamlit."""
    if not fig.axes:
        return
    ax = fig.axes[0]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_autoscale_on(False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def plot_modes_grid(Vn, niveles, T, title_suptitle, include_base_minus1=False, ncols=6):
    """
    ✅ Siempre máximo 6 por fila.
    ✅ Si n_modos < 6, ncols se reduce a n_modos (sin blancos).
    """
    Vn = np.asarray(Vn, dtype=float)
    T  = np.asarray(T,  dtype=float).ravel()

    # Normalizar por modo (para que siempre quepa)
    den = np.max(np.abs(Vn), axis=0)
    den = np.where(den == 0, 1.0, den)
    Vplot = Vn / den

    n_modos = int(Vplot.shape[1])
    if n_modos <= 0:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.text(0.5, 0.5, "Sin modos para graficar", ha="center", va="center",
                color=COLOR_TEXT, fontsize=12, path_effects=halo2)
        ax.axis("off")
        return fig

    ncols = int(max(1, min(int(ncols), n_modos)))   # ✅ ajusta si hay <6
    nrows = int(np.ceil(n_modos / ncols))

    # tamaño base por celda
    fig_w = 1.75 * ncols
    fig_h = 3.9  * nrows

    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=True)
    fig.patch.set_facecolor(BG)

    # axs a lista plana
    axs_list = np.array(axs).ravel().tolist() if isinstance(axs, np.ndarray) else [axs]
    for ax in axs_list:
        ax.set_facecolor(BG)

    fs_title = 7.5

    for i in range(n_modos):
        ax = axs_list[i]

        if include_base_minus1:
            modo = np.concatenate([[0.0], Vplot[:, i]])  # agrega un punto para -1
            y = niveles
        else:
            modo = np.insert(Vplot[:, i], 0, 0.0)        # base 0
            y = niveles

        ax.plot(modo,  y, "-o", color=COLOR_MODO, lw=1.05, ms=2.5)
        ax.plot(-modo, y, "--o", color=COLOR_INV,  lw=0.90, ms=2.2, alpha=0.95)
        ax.plot(np.zeros_like(y), y, "-", color=COLOR_STRUCT, lw=0.75, alpha=0.85)

        ax.set_title(f"Modo {i+1}\nT={T[i]:.3f} s", color=COLOR_TEXT, fontsize=fs_title, path_effects=halo2)
        ax.tick_params(colors=COLOR_TEXT, labelsize=8)
        ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-1.1, 1.1)

        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["-1", "0", "1"], color=COLOR_TEXT, fontsize=8)

    # ocultar ejes vacíos
    for j in range(n_modos, len(axs_list)):
        axs_list[j].axis("off")

    # ylabel solo primera columna
    for r in range(nrows):
        axs_list[r*ncols].set_ylabel("Altura [m]", color=COLOR_TEXT)

    fig.suptitle(title_suptitle, color=COLOR_TEXT, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.965])
    _freeze_axes_limits(fig)
    return fig

def _force_same_height_modes(figA, figB, nA, nB, ncols=6, w_cell=1.75, h_row=3.9, h_min=4.9):
    """
    Fuerza que FIJA y AISLADA tengan la MISMA altura visual (misma cantidad de filas).
    """
    ncolsA = max(1, min(ncols, int(max(nA, 1))))
    ncolsB = max(1, min(ncols, int(max(nB, 1))))
    nrowsA = int(np.ceil(max(nA, 1) / ncolsA))
    nrowsB = int(np.ceil(max(nB, 1) / ncolsB))
    nrows  = int(max(nrowsA, nrowsB, 1))

    # ancho lo dejamos igual en ambas (máximo 6 columnas)
    W = float(w_cell * ncols)
    H = float(max(h_min, h_row * nrows))

    figA.set_size_inches(W, H, forward=True)
    figB.set_size_inches(W, H, forward=True)

    figA.tight_layout(rect=[0, 0.01, 1, 0.965])
    figB.tight_layout(rect=[0, 0.01, 1, 0.965])

    _freeze_axes_limits(figA)
    _freeze_axes_limits(figB)
    return figA, figB

def extraer_rigideces_para_esquema(K: np.ndarray, tiene_base: bool):
    K = np.asarray(K, dtype=float)
    n = K.shape[0]

    if not tiene_base:
        if n == 1:
            return None, np.array([abs(K[0, 0])], dtype=float)

        k_story = np.zeros(n, dtype=float)
        for i in range(1, n):
            k_story[i] = abs(K[i-1, i])
        k_story[0] = abs(K[0, 0] - k_story[1])
        return None, k_story

    else:
        if n < 2:
            raise ValueError("K aislado debe tener al menos 2 DOF (base y un piso).")

        n_pisos = n - 1
        k_story = np.zeros(n_pisos, dtype=float)

        k_story[0] = abs(K[0, 1])
        for i in range(2, n):
            k_story[i-1] = abs(K[i-1, i])

        k_iso = abs(K[0, 0] + K[0, 1])
        return float(k_iso), k_story

def plot_modelo_condensado_fijo(K_fix, M_fix, niveles_fix, pisos_y):
    n_pisos_fix = len(pisos_y)

    fs = max(5.0, 9.5 - 0.12*n_pisos_fix)
    lw_ed = max(0.9, 2.4 - 0.05*n_pisos_fix)
    ms = max(4.0, 10.0 - 0.18*n_pisos_fix)

    fig, ax = plt.subplots(figsize=(5.1, 6.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y = np.asarray(niveles_fix, dtype=float)
    _, k_fix_plot = extraer_rigideces_para_esquema(K_fix, tiene_base=False)

    x_center = 0.0
    xk_txt = x_center - 0.12   # ← IZQUIERDA del eje
    xm = x_center
    xm_txt = x_center + 0.10   # → DERECHA del eje

    XMAX = max(abs(xk_txt) + 0.35, abs(xm_txt) + 0.55)

    # ✅ ÚNICA línea azul (edificio)
    ax.plot([x_center, x_center], [0, np.max(y)], color=COLOR_SPR, lw=lw_ed)

    # Base corta
    base_half = 0.12
    ax.plot([x_center-base_half, x_center+base_half], [0, 0],
            color=COLOR_BASE, lw=5, solid_capstyle="round")
    ax.text(x_center+base_half+0.06, 0, "Base fija",
            va="center", fontsize=fs, color=COLOR_BASE, path_effects=HALO)

    # Rigideces (texto a la IZQUIERDA)
    for i in range(n_pisos_fix):
        ymid = 0.5*(y[i] + y[i+1])
        ax.text(
            xk_txt, ymid,
            f"$K_{{{i+1}}}={k_fix_plot[i]:.1f}$ Tf/m",
            color=COLOR_LABEL_SPR, fontsize=fs,
            va="center", ha="right", path_effects=HALO
        )

    # Masas (bolita + texto a la derecha)
    for i in range(1, n_pisos_fix+1):
        ax.plot(xm, y[i], "o", color=COLOR_MASS, markersize=ms)
        ax.text(
            xm_txt, y[i],
            f"$M_{{{i}}}={M_fix[i-1,i-1]:.3f}$ Tf·s²/m",
            color=COLOR_LABEL_MASS, fontsize=fs,
            va="center", ha="left", path_effects=HALO
        )

    ax.set_title("Modelo Condensado FIJO", color=COLOR_TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel("Altura [m]", color=COLOR_TEXT)
    ax.set_xlim(-XMAX, XMAX)
    ax.set_ylim(0, np.max(y)+1)
    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.35)
    ax.tick_params(colors=COLOR_TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])

    fig.tight_layout()
    _freeze_axes_limits(fig)
    return fig

def plot_modelo_condensado_aislado(K_cond_ais, M_cond_ais, pisos_y):
    n_pisos = len(pisos_y)

    fs = max(5.0, 9.5 - 0.12*n_pisos)
    lw_ed = max(0.9, 2.4 - 0.05*n_pisos)
    ms = max(4.0, 10.0 - 0.18*n_pisos)

    fig, ax = plt.subplots(figsize=(5.1, 6.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    pisos_y = np.asarray(pisos_y, dtype=float).ravel()
    y_top = float(np.max(pisos_y)) if len(pisos_y) else 0.0

    # ✅ Niveles del aislado: [-1, 0, y1, y2, ...]
    yline = np.concatenate([[-1.0], [0.0], pisos_y])

    # Rigideces equivalentes para esquema
    k_iso_plot, k_story_ais_plot = extraer_rigideces_para_esquema(K_cond_ais, tiene_base=True)

    # ----------------- Geometría / posiciones -----------------
    x_center = 0.0

    # ✅ más cerca del eje (no tan a la izquierda)
    xk_txt  = x_center - 0.12   # antes -0.18
    xm      = x_center
    xm_txt  = x_center + 0.10

    # Límite horizontal automático (para que no se corte texto)
    XMAX = max(abs(xk_txt) + 0.45, abs(xm_txt) + 0.65)

    # ✅ ÚNICA línea azul del edificio: desde -1 hasta el techo
    ax.plot([x_center, x_center], [-1.0, y_top], color=COLOR_SPR, lw=lw_ed)

    # ✅ Base fija corta en y=-1
    base_half = 0.12
    ax.plot([x_center-base_half, x_center+base_half], [-1.0, -1.0],
            color=COLOR_BASE, lw=5, solid_capstyle="round")
    ax.text(x_center+base_half+0.06, -1.0, "Base fija",
            va="center", fontsize=fs, color=COLOR_BASE, path_effects=HALO)

    # ----------------- Textos K (a la izquierda del eje) -----------------
    # K_ais en el medio del aislador (-1 a 0)
    ax.text(
        xk_txt, -0.5,
        f"$K_{{ais}}={k_iso_plot:.1f}$ Tf/m",
        color=COLOR_LABEL_SPR, fontsize=fs,
        va="center", ha="right", path_effects=HALO
    )

    # K1..Kn en los entrepisos 0->y1, y1->y2, ...
    # (yline = [-1,0,y1,y2...], entonces entrepisos estructurales empiezan en i=2)
    for i in range(2, len(yline)):
        ymid = 0.5 * (yline[i-1] + yline[i])
        k_i = float(k_story_ais_plot[i-2])  # k1 corresponde a 0->y1
        ax.text(
            xk_txt, ymid,
            f"$K_{{{i-1}}}={k_i:.1f}$ Tf/m",
            color=COLOR_LABEL_SPR, fontsize=fs,
            va="center", ha="right", path_effects=HALO
        )

    # ----------------- Masas (bolita sobre eje + texto a la derecha) -----------------
    # M0 en y=0
    ax.plot(xm, 0.0, "o", color=COLOR_MASS, markersize=ms)
    ax.text(
        xm_txt, 0.0,
        f"$M_{{0}}={M_cond_ais[0,0]:.3f}$ Tf·s²/m",
        color=COLOR_LABEL_MASS, fontsize=fs,
        va="center", ha="left", path_effects=HALO
    )

    # M1..Mn en pisos
    for i, yv in enumerate(pisos_y, start=1):
        ax.plot(xm, float(yv), "o", color=COLOR_MASS, markersize=ms)
        ax.text(
            xm_txt, float(yv),
            f"$M_{{{i}}}={M_cond_ais[i,i]:.3f}$ Tf·s²/m",
            color=COLOR_LABEL_MASS, fontsize=fs,
            va="center", ha="left", path_effects=HALO
        )

    # ----------------- Estilo ejes -----------------
    ax.set_title("Modelo Condensado AISLADO", color=COLOR_TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel("Altura [m]", color=COLOR_TEXT)

    ax.set_xlim(-XMAX, XMAX)
    ax.set_ylim(-1.5, y_top + 1.0)     # ✅ límite inferior basado en -1
    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.35)
    ax.tick_params(colors=COLOR_TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])

    fig.tight_layout()
    _freeze_axes_limits(fig)
    return fig

# -----------------------------------------------------------------
# 5A) GENERAR MATRICES AISLADAS (SIEMPRE) + GUARDAR
# -----------------------------------------------------------------
try:
    gdl_map_libre, total_dofs_libre = generar_gdl_map_reducido_libre(nodes)

    elements_libre = []
    for n1, n2, etipo in element_node_pairs:
        node_i = nodes[n1]
        node_j = nodes[n2]
        E_el = propiedades[etipo]["E"]
        I_el = propiedades[etipo]["I"]
        A_el = propiedades[etipo]["A"]
        elements_libre.append(
            ElementReducidoLibre(node_i, node_j, E_el, I_el, A_el, gdl_map_libre)
        )

    K_global_libre = assemble_global_stiffness_reducido_libre(elements_libre, total_dofs_libre)

    gdl_pp = sorted({dof for (_nid, gtipo), dof in gdl_map_libre.items() if gtipo == "vx" and dof is not None})
    gdl_ss = sorted({dof for (_nid, gtipo), dof in gdl_map_libre.items() if gtipo == "theta" and dof is not None})

    kpp = K_global_libre[np.ix_(gdl_pp, gdl_pp)]
    kss = K_global_libre[np.ix_(gdl_ss, gdl_ss)]
    kps = K_global_libre[np.ix_(gdl_pp, gdl_ss)]
    ksp = K_global_libre[np.ix_(gdl_ss, gdl_pp)]

    K_vx_nodo = kpp - kps @ solve(kss, ksp)

    nodes_arr = np.array(nodes, dtype=float)
    pisos_y = np.sort(np.unique(nodes_arr[:, 1][nodes_arr[:, 1] > 0]))
    niveles_y = np.insert(pisos_y, 0, 0.0)  # [0, y1, y2, ...]
    niveles_y_r = [round(float(y), 6) for y in niveles_y]
    y_to_col = {y: i for i, y in enumerate(niveles_y_r)}

    dof_to_row = {dof: i for i, dof in enumerate(gdl_pp)}
    T_pp_ais = np.zeros((len(gdl_pp), len(niveles_y_r)))

    for (x, y, nid) in nodes:
        dof_vx = gdl_map_libre.get((nid, "vx"))
        if dof_vx is None:
            continue
        y_key = round(float(y), 6)
        if y_key not in y_to_col:
            continue
        row = dof_to_row.get(dof_vx, None)
        if row is None:
            continue
        col = y_to_col[y_key]
        T_pp_ais[row, col] = 1.0

    K_cond_ais = T_pp_ais.T @ K_vx_nodo @ T_pp_ais
    K_cond_ais[0, 0] += keff_1ais * n_aisladores

    M_cond_ais = calcular_matriz_masas_con_aislador(
        nodes,
        element_node_pairs,
        propiedades,
        peso_especifico=st.session_state["peso_especifico"],
        sobrecarga_muerta=st.session_state["sobrecarga_muerta"],
    )

    st.session_state["K_cond_ais"] = np.array(K_cond_ais, copy=True)
    st.session_state["M_cond_ais"] = np.array(M_cond_ais, copy=True)

except Exception as e:
    st.error(f"❌ Error al generar matrices AISLADAS: {e}")
    st.stop()

# -----------------------------------------------------------------
# 5B) MATRICES (EXPANDERS)
# -----------------------------------------------------------------
st.subheader("📘 Matrices condensadas (comparación)")
colM1, colM2 = st.columns([1, 1], gap="large")

with colM1:
    with st.expander("🧱 Matrices del modelo FIJO (K_cond, M_cond)", expanded=False):
        st.markdown(f"**Rigidez condensada FIJA** (dim = {np.asarray(K_fix).shape[0]}×{np.asarray(K_fix).shape[1]}):")
        st.dataframe(pd.DataFrame(np.round(np.asarray(K_fix, float), 3)), use_container_width=True)
        st.markdown(f"**Masa condensada FIJA** (dim = {np.asarray(M_fix).shape[0]}×{np.asarray(M_fix).shape[1]}):")
        st.dataframe(pd.DataFrame(np.round(np.asarray(M_fix, float), 5)), use_container_width=True)

with colM2:
    with st.expander("🟩 Matrices del modelo AISLADO (K_cond_ais, M_cond_ais)", expanded=False):
        st.markdown(f"**Rigidez condensada AISLADA** (dim = {K_cond_ais.shape[0]}×{K_cond_ais.shape[1]}):")
        st.dataframe(pd.DataFrame(np.round(K_cond_ais, 3)), use_container_width=True)
        st.markdown(f"**Masa condensada AISLADA** (dim = {M_cond_ais.shape[0]}×{M_cond_ais.shape[1]}):")
        st.dataframe(pd.DataFrame(np.round(M_cond_ais, 5)), use_container_width=True)

# -----------------------------------------------------------------
# 5C) ANÁLISIS MODAL (SIMÉTRICO)
# -----------------------------------------------------------------
st.subheader("📊 Análisis modal (comparación simétrica)")

niveles_fix = np.insert(pisos_y, 0, 0.0)               # [0, y1, y2...]
niveles_ais = np.concatenate([[-1.0], [0.0], pisos_y]) # [-1, 0, y1...]

w_fix, T_fix, f_fix, Vn_fix = modal_props(np.asarray(K_fix, float), np.asarray(M_fix, float))
w_ais, T_ais, f_ais, Vn_ais = modal_props(K_cond_ais, M_cond_ais)

# Guardar (como venías usando)
st.session_state["w_sin"] = w_fix
st.session_state["T_sin"] = T_fix
st.session_state["v_norm_sin"] = Vn_fix

st.session_state["w_ais"] = w_ais
st.session_state["T_ais"] = T_ais
st.session_state["v_norm_ais"] = Vn_ais

n_modos_fix = int(Vn_fix.shape[1])
n_modos_ais = int(Vn_ais.shape[1])

# -----------------------------------------------------------------
# ✅ FIGURA 1: MODOS (6 por fila y misma altura en FIJA/AISLADA)
# -----------------------------------------------------------------
fig_fix_modes = plot_modes_grid(
    Vn_fix, niveles_fix, T_fix,
    "Modos de Vibración – Estructura FIJA",
    include_base_minus1=False,
    ncols=6
)

fig_ais_modes = plot_modes_grid(
    Vn_ais, niveles_ais, T_ais,
    "Modos de Vibración – Estructura AISLADA",
    include_base_minus1=True,
    ncols=6
)

fig_fix_modes, fig_ais_modes = _force_same_height_modes(
    fig_fix_modes, fig_ais_modes,
    nA=n_modos_fix, nB=n_modos_ais,
    ncols=6
)

# -----------------------------------------------------------------
# ✅ FIGURA 2: ESQUEMAS (misma altura, sin corrimientos)
# -----------------------------------------------------------------
fig_fix_scheme = plot_modelo_condensado_fijo(np.asarray(K_fix, float), np.asarray(M_fix, float), niveles_fix, pisos_y)
fig_ais_scheme = plot_modelo_condensado_aislado(K_cond_ais, M_cond_ais, pisos_y)

# -----------------------------------------------------------------
# Render simétrico
# -----------------------------------------------------------------
colL, colR = st.columns([1, 1], gap="large")

with colL:
    with st.container(border=True):
        st.subheader("1️⃣ Frecuencias y períodos – FIJA")
        tabla_fix = pd.DataFrame({
            "Modo": np.arange(1, len(f_fix) + 1),
            "f [Hz]": np.round(f_fix, 5),
            "T [s]": np.round(T_fix, 5),
        })
        st.dataframe(tabla_fix, hide_index=True, use_container_width=True, height=170)

    with st.container(border=True):
        st.subheader("2️⃣ Modos de vibración – FIJA")
        st.pyplot(fig_fix_modes, use_container_width=True)
        plt.close(fig_fix_modes)

    with st.container(border=True):
        st.subheader("3️⃣ Esquema del modelo – FIJA")
        st.pyplot(fig_fix_scheme, use_container_width=True)
        plt.close(fig_fix_scheme)

with colR:
    with st.container(border=True):
        st.subheader("1️⃣ Frecuencias y períodos – AISLADA")
        tabla_ais = pd.DataFrame({
            "Modo": np.arange(1, len(f_ais) + 1),
            "f [Hz]": np.round(f_ais, 5),
            "T [s]": np.round(T_ais, 5),
        })
        st.dataframe(tabla_ais, hide_index=True, use_container_width=True, height=170)

    with st.container(border=True):
        st.subheader("2️⃣ Modos de vibración – AISLADA")
        st.pyplot(fig_ais_modes, use_container_width=True)
        plt.close(fig_ais_modes)

    with st.container(border=True):
        st.subheader("3️⃣ Esquema del modelo – AISLADA")
        st.pyplot(fig_ais_scheme, use_container_width=True)
        plt.close(fig_ais_scheme)

st.success("✅ Bloque 5 listo: MODOS (máx 6 por fila, auto-ajuste sin blancos, misma altura FIJA/AISLADA) + ESQUEMAS (misma altura y sin corrimientos).")

# =============================================================================
# === BLOQUE 6: ANÁLISIS DINÁMICO (NEWMARK-β) SIMÉTRICO =======================
# === IZQ: FIJA | DER: AISLADA ===============================================
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st

st.subheader("⚙️ Análisis dinámico lineal (Newmark-β) – comparación simétrica")

# -----------------------------------------------------------------
# ✅ PRERREQUISITOS (compacto)
# -----------------------------------------------------------------
req_keys = [
    "K_cond", "M_cond",
    "K_cond_ais", "M_cond_ais",
    "ag_filt", "t_ag", "dt",
]
missing = [k for k in req_keys if k not in st.session_state]
if missing:
    st.info(f"⚙️ Faltan en session_state: {', '.join(missing)}. Ejecuta bloques anteriores.")
    st.stop()

def get_fun(name):
    f = st.session_state.get(name, None)
    if f is None and name in globals():
        f = globals()[name]
    return f

newmark = get_fun("newmark")
graficar_respuesta_por_piso = get_fun("graficar_respuesta_por_piso")

if newmark is None:
    st.error("❌ No se encontró la función `newmark` (ni en globals ni en session_state).")
    st.stop()

if graficar_respuesta_por_piso is None:
    st.error("❌ No se encontró `graficar_respuesta_por_piso` (ni en globals ni en session_state).")
    st.stop()

# -----------------------------------------------------------------
# ✅ Lectura de datos
# -----------------------------------------------------------------
K_fix = np.array(st.session_state["K_cond"], dtype=float)
M_fix = np.array(st.session_state["M_cond"], dtype=float)

K_ais = np.array(st.session_state["K_cond_ais"], dtype=float)
M_ais = np.array(st.session_state["M_cond_ais"], dtype=float)

ag_filt = np.asarray(st.session_state["ag_filt"], float).ravel()
t_ag    = np.asarray(st.session_state["t_ag"], float).ravel()
dt      = float(st.session_state["dt"])

# -----------------------------------------------------------------
# ✅ ζ: NO volver a pedirlo si ya se ingresó en el BLOQUE 4
#    - En tu bloque 4 lo guardaste con key="zeta_ray"
# -----------------------------------------------------------------
if "zeta_ray" in st.session_state and st.session_state["zeta_ray"] is not None:
    zeta = float(st.session_state["zeta_ray"])
    st.caption(f"Usando amortiguamiento modal objetivo ζ = **{zeta:.3f}** (ingresado en Bloque 4).")
else:
    zeta = st.number_input("Amortiguamiento modal ζ", 0.0, 0.2, 0.05, 0.005, key="zeta_ray_dyn")

# -----------------------------------------------------------------
# ✅ Newmark (promedio constante)
# -----------------------------------------------------------------
gamma_n = 0.5
beta_n  = 0.25

# Duración total (misma lógica tuya)
t_total = float(t_ag[-1]) + 2.0
n_total = int(np.floor(t_total / dt)) + 1
t = np.linspace(0.0, (n_total - 1) * dt, n_total)

# Extender ag a la duración total
ag_ext = np.pad(ag_filt, (0, max(0, n_total - len(ag_filt))), mode="constant")[:n_total]

# -----------------------------------------------------------------
# 🧠 Utilidades
# -----------------------------------------------------------------
def rayleigh_from_w(w_in, zeta):
    """
    w_in: rad/s (si max>10) o Hz (si max<10)
    """
    w_in = np.asarray(w_in, dtype=float).ravel()
    if len(w_in) < 2:
        raise ValueError("Se requieren al menos 2 frecuencias naturales para Rayleigh.")
    w = (2*np.pi*w_in) if np.max(w_in) < 10.0 else w_in
    w1, w2 = float(w[0]), float(w[1])
    A = np.array([[1/(2*w1), w1/2],
                  [1/(2*w2), w2/2]], dtype=float)
    b = np.array([zeta, zeta], dtype=float)
    alpha, beta = np.linalg.solve(A, b)
    return float(alpha), float(beta)

def ensure_2d(u, v, a):
    if u.ndim == 1:
        u = u[np.newaxis, :]
        v = v[np.newaxis, :]
        a = a[np.newaxis, :]
    return u, v, a

def modal_w(K, M):
    A = np.linalg.inv(M) @ K
    w2, _ = np.linalg.eig(A)
    w2 = np.sort(np.real(w2))
    w2[w2 < 0] = 0
    return np.sqrt(w2)

# Usar w ya guardadas en bloque 5 (o calcular rápido)
w_fix = st.session_state.get("w_sin", None)
if w_fix is None:
    w_fix = modal_w(K_fix, M_fix)

w_ais = st.session_state.get("w_ais", None)
if w_ais is None:
    w_ais = modal_w(K_ais, M_ais)

# Alturas reales (para graficar por piso)
nodes = st.session_state.get("nodes", None)
if nodes is not None:
    nodes_arr = np.array(nodes, dtype=float)
    pisos_y_empotrado = np.sort(np.unique(nodes_arr[:, 1][nodes_arr[:, 1] > 0]))
else:
    pisos_y_empotrado = None  # fallback dentro del plot

# -----------------------------------------------------------------
# 📊 Layout simétrico
# -----------------------------------------------------------------
colL, colR = st.columns([1, 1], gap="large")

# ========================= IZQUIERDA (FIJA) =========================
with colL:
    with st.container(border=True):
        st.markdown("### 🧱 FIJA – Rayleigh + Newmark")

        alpha_fix, beta_fix = rayleigh_from_w(w_fix, zeta)
        C_fix = alpha_fix * M_fix + beta_fix * K_fix

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("α", f"{alpha_fix:.3e}", "1/s")
        m2.metric("β", f"{beta_fix:.3e}", "s")
        m3.metric("ζ", f"{zeta:.3f}", "")
        m4.metric("Duración", f"{t_total:.2f}", "s")

        r_fix = np.ones((M_fix.shape[0], 1))
        P_fix = -(M_fix @ r_fix) @ ag_ext[np.newaxis, :]

        U0 = np.zeros(K_fix.shape[0])
        V0 = np.zeros(K_fix.shape[0])

        u_fix, v_fix_t, a_fix_t = newmark(
            M_fix, C_fix, K_fix, U0, V0, dt, P_fix, gamma=gamma_n, beta=beta_n
        )
        u_fix, v_fix_t, a_fix_t = ensure_2d(u_fix, v_fix_t, a_fix_t)

        # Guardar (mismas claves)
        st.session_state["u_t"]   = u_fix
        st.session_state["v_t"]   = v_fix_t
        st.session_state["a_t"]   = a_fix_t
        st.session_state["t_fix"] = t
        st.session_state["C_fix"] = C_fix

    with st.container(border=True):
        st.markdown("### 📈 FIJA – Respuestas por piso")

        if pisos_y_empotrado is None:
            pisos_y_empotrado = np.arange(u_fix.shape[0], dtype=float)

        st.session_state["alturas"] = np.array(pisos_y_empotrado, dtype=float)

        n_gdl_fix = int(u_fix.shape[0])
        # 1 columna, igual que venías
        for idx in range(n_gdl_fix):
            piso_real = pisos_y_empotrado[idx] if idx < len(pisos_y_empotrado) else idx + 1
            st.markdown(f"**Piso {idx + 1}**")
            graficar_respuesta_por_piso(
                t,
                u_fix[[idx], :],
                v_fix_t[[idx], :],
                a_fix_t[[idx], :],
                [piso_real],
                t_total,
                nombre_piso=str(idx + 1),
            )

# ========================= DERECHA (AISLADA) =========================
with colR:
    with st.container(border=True):
        st.markdown("### 🟩 AISLADA – Rayleigh + Newmark")

        alpha_ais, beta_ais = rayleigh_from_w(w_ais, zeta)
        C_ais = alpha_ais * M_ais + beta_ais * K_ais

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("α", f"{alpha_ais:.3e}", "1/s")
        m2.metric("β", f"{beta_ais:.3e}", "s")
        m3.metric("ζ", f"{zeta:.3f}", "")
        m4.metric("Duración", f"{t_total:.2f}", "s")

        r_ais = np.ones((M_ais.shape[0], 1))
        P_ais = -(M_ais @ r_ais) @ ag_ext[np.newaxis, :]

        U0 = np.zeros(K_ais.shape[0])
        V0 = np.zeros(K_ais.shape[0])

        u_ais, v_ais_t, a_ais_t = newmark(
            M_ais, C_ais, K_ais, U0, V0, dt, P_ais, gamma=gamma_n, beta=beta_n
        )
        u_ais, v_ais_t, a_ais_t = ensure_2d(u_ais, v_ais_t, a_ais_t)

        st.session_state["u_t_ais"] = u_ais
        st.session_state["v_t_ais"] = v_ais_t
        st.session_state["a_t_ais"] = a_ais_t
        st.session_state["t_ais"]   = t
        st.session_state["C_ais"]   = C_ais

    with st.container(border=True):
        st.markdown("### 📈 AISLADA – Respuestas por piso (incluye aislador)")

        # alturas: [-1 (base fija), 0 (aislador), pisos...]
        alturas_reales = st.session_state.get("alturas", None)
        if alturas_reales is None:
            if nodes is not None:
                alturas_reales = np.array(sorted(set([yy for (_x, yy, _id) in nodes if float(yy) > 0])), dtype=float)
            else:
                alturas_reales = np.arange(max(u_ais.shape[0] - 1, 0), dtype=float)

        alturas_ais = np.concatenate([[-1.0], [0.0], np.asarray(alturas_reales, float)])
        st.session_state["alturas_ais"] = alturas_ais

        n_gdl_ais = int(u_ais.shape[0])
        for idx in range(n_gdl_ais):
            # idx=0 => aislador (nivel 0)
            if idx == 0:
                st.markdown("**Aislador (Nivel 0)**")
                nombre = "0"
                h = 0.0
            else:
                st.markdown(f"**Piso {idx}**")
                nombre = str(idx)
                h = float(alturas_ais[idx + 1]) if (idx + 1) < len(alturas_ais) else float(idx)

            graficar_respuesta_por_piso(
                t,
                u_ais[[idx], :],
                v_ais_t[[idx], :],
                a_ais_t[[idx], :],
                [h],
                t_total,
                nombre_piso=nombre,
            )

st.success("✅ Bloque 6 listo: Newmark-β simétrico (sin repetir ζ si ya se ingresó en Bloque 4).")

# =============================================================================
# === BLOQUE 7: ESPECTRO NEC24 (izq) + HISTÉRESIS (der) =======================
# =============================================================================
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

st.markdown("## 📌 Espectro NEC-24 + Histéresis del aislador (comparación)")

# ----------------------- Estilos (tus colores) -----------------------
BG         = "#2B3141"
COLOR_TEXT = "#E8EDF2"
COLOR_GRID = "#5B657A"
HALO = [pe.withStroke(linewidth=2.4, foreground=BG), pe.Normal()]

# Paleta espectro
COLOR_ELAST    = "#A8D5FF"
COLOR_INELAST  = "#F2A6A0"
COLOR_MARK_FIX = "#FFE6A3"
COLOR_MARK_AIS = "#77DD77"
COLOR_GUIDE    = "#7A8498"
LEG_FACE       = "#3A4050"
LEG_EDGE       = "#A7B1C5"

# Paleta histéresis
COLOR_LINE1 = "#C79BFF"

# -----------------------------------------------------------------
# ✅ PRERREQUISITOS (ordenado y sin loops largos)
# -----------------------------------------------------------------
def need(key, label=None):
    v = st.session_state.get(key, None)
    return (label or f"st.session_state['{key}']") if v is None else None

faltantes = []

# Espectro guardado
for k in ["T_spec", "Sa_elast_spec", "Sa_inelas_spec", "T_final_spec", "R_spec"]:
    m = need(k)
    if m: faltantes.append(m)

# Periodos (para marcar puntos)
for k in ["T_sin", "T_ais"]:
    m = need(k)
    if m: faltantes.append(m)

# Matrices aisladas (para histéresis)
for k in ["M_cond_ais", "K_cond_ais", "dt", "ag_filt"]:
    m = need(k)
    if m: faltantes.append(m)

# Parámetros bilineales (pueden estar en session_state o globals)
def has_param(name):
    return (name in st.session_state) or (name in globals())

for v in ["k_inicial_1ais", "k_post_1ais", "yield_1ais", "c_1ais"]:
    if not has_param(v):
        faltantes.append(v)

# Función NL (NO se define aquí)
if "newmark_nl_base_bilinear" not in globals():
    faltantes.append("newmark_nl_base_bilinear (definida en tu proyecto)")

if faltantes:
    st.warning("⚠️ Faltan variables para este bloque: " + ", ".join(faltantes))
    st.stop()

# -----------------------------------------------------------------
# Layout: Izq (espectro) | Der (histéresis) -> MISMO TAMAÑO
# -----------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")
FIG_W, FIG_H = 7.2, 4.8

# =============================================================================
# IZQUIERDA: ESPECTRO NEC24 + PUNTOS FIJO / AISLADO
# =============================================================================
with col_left:
    with st.container(border=True):
        st.subheader("📈 Espectro NEC-24 + puntos (FIJA vs AISLADA)")

        T_plot   = np.asarray(st.session_state["T_spec"], dtype=float).ravel()
        Sa_el    = np.asarray(st.session_state["Sa_elast_spec"], dtype=float).ravel()
        Sa_inel  = np.asarray(st.session_state["Sa_inelas_spec"], dtype=float).ravel()
        T_final  = float(st.session_state["T_final_spec"])
        R_spec   = float(st.session_state["R_spec"])

        if not (len(T_plot) == len(Sa_el) == len(Sa_inel)):
            st.error(
                f"❌ Dimensiones no coinciden: T_spec={T_plot.shape}, "
                f"Sa_elast={Sa_el.shape}, Sa_inelas={Sa_inel.shape}."
            )
            st.stop()

        T1_fix = float(np.asarray(st.session_state["T_sin"], dtype=float).ravel()[0])
        T1_ais = float(np.asarray(st.session_state["T_ais"], dtype=float).ravel()[0])

        Sa_Tfix = float(np.interp(T1_fix, T_plot, Sa_el))
        Sa_Tais = float(np.interp(T1_ais, T_plot, Sa_el))

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        ax.plot(T_plot, Sa_el,   lw=2.4, color=COLOR_ELAST,   label="Espectro Elástico")
        ax.plot(T_plot, Sa_inel, lw=2.2, color=COLOR_INELAST, linestyle="--",
                label=f"Espectro Inelástico (R={R_spec:g})")

        # Guías + punto FIJA
        ax.plot([T1_fix, T1_fix], [0, Sa_Tfix], color=COLOR_GUIDE, linestyle=":", lw=1.3)
        ax.plot([0, T1_fix], [Sa_Tfix, Sa_Tfix], color=COLOR_GUIDE, linestyle=":", lw=1.3)
        ax.plot(T1_fix, Sa_Tfix, "o", ms=7, mfc=COLOR_MARK_FIX, mec="none",
                label=f"FIJA: T₁={T1_fix:.3f} s | Sa={Sa_Tfix:.3f} g")

        # Guías + punto AISLADA
        ax.plot([T1_ais, T1_ais], [0, Sa_Tais], color=COLOR_GUIDE, linestyle=":", lw=1.3, alpha=0.85)
        ax.plot([0, T1_ais], [Sa_Tais, Sa_Tais], color=COLOR_GUIDE, linestyle=":", lw=1.3, alpha=0.85)
        ax.plot(T1_ais, Sa_Tais, "o", ms=7, mfc=COLOR_MARK_AIS, mec="none",
                label=f"AISLADA: T₁={T1_ais:.3f} s | Sa={Sa_Tais:.3f} g")

        ax.set_xlabel("Período T [s]", color=COLOR_TEXT)
        ax.set_ylabel("Aceleración espectral Sa [g]", color=COLOR_TEXT)
        ax.set_title("Espectro de Respuesta Elástico e Inelástico - NEC 24",
                     color=COLOR_TEXT, fontweight="bold")

        ax.set_xlim(0, T_final)
        ax.set_ylim(bottom=0)
        ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
        ax.tick_params(colors=COLOR_TEXT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        leg = ax.legend(facecolor=LEG_FACE, edgecolor=LEG_EDGE, framealpha=0.95, loc="best")
        for txt in leg.get_texts():
            txt.set_color(COLOR_TEXT)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

# =============================================================================
# DERECHA: HISTÉRESIS REAL DEL AISLADOR (SOLO F–u)
# =============================================================================
with col_right:
    with st.container(border=True):
        st.subheader("🟣 Histéresis real del aislador (bilineal)")

        dt     = float(st.session_state["dt"])
        ag_g   = np.asarray(st.session_state["ag_filt"], dtype=float).ravel()

        M_ais  = np.array(st.session_state["M_cond_ais"], copy=True, dtype=float)
        K_ais  = np.array(st.session_state["K_cond_ais"], copy=True, dtype=float)

        # Si existe C_ais (del bloque Newmark lineal), úsalo; si no, cero.
        C_used = st.session_state.get("C_ais", None)
        C_used = np.zeros_like(M_ais) if C_used is None else np.array(C_used, copy=True, dtype=float)

        def _get(name):
            return float(st.session_state[name]) if name in st.session_state else float(globals()[name])

        k0    = _get("k_inicial_1ais")
        kp    = _get("k_post_1ais")
        Fy    = _get("yield_1ais")
        c_iso = _get("c_1ais")

        U_nl, V_nl, A_nl, Fiso_hist, Fhyst_hist, Ehyst = newmark_nl_base_bilinear(
            M=M_ais, C=C_used, K=K_ais,
            dt=dt, ag_g=ag_g,
            k0=k0, kp=kp, Fy=Fy, c_iso=c_iso,
            gamma=0.5, beta=0.25,
            newton_tol=1e-7, newton_maxit=30
        )

        # Guardar por si lo necesitas después
        st.session_state["U_nl"]      = U_nl
        st.session_state["Fiso_hist"] = Fiso_hist

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        ax.plot(U_nl[0, :], Fiso_hist, color=COLOR_LINE1, lw=1.2)

        ax.set_xlabel("Desplazamiento base aislada u₀ [m]", color=COLOR_TEXT, fontsize=9)
        ax.set_ylabel("Fuerza total del aislador [Tf]", color=COLOR_TEXT, fontsize=9)
        ax.set_title("Curva histerética real del aislador (Newmark–Newton)",
                     color=COLOR_TEXT, fontsize=12, fontweight="semibold")

        ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.35)
        ax.tick_params(colors=COLOR_TEXT, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLOR_GRID)
        ax.spines["bottom"].set_color(COLOR_GRID)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

st.success("✅ Listo: Espectro (FIJA vs AISLADA) + Histéresis (solo F–u), mismo tamaño.")

# =============================================================================
# === BLOQUE 8: CORTANTES POR PISO (RSA vs THA) – FIJA vs AISLADA =============
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

st.markdown("## 🧱 Cortantes por piso – Modal espectral (RSA) vs Tiempo historia (THA)")

# ----------------------- Estilos -----------------------
BG            = "#2B3141"
COLOR_TEXT    = "#E8EDF2"
COLOR_GRID    = "#5B657A"
COLOR_FIX     = "#A8D5FF"
COLOR_AIS     = "#77DD77"
COLOR_GUIDE   = "#FFDFA0"
HALO = [pe.withStroke(linewidth=2.4, foreground=BG), pe.Normal()]

# ----------------------- Helpers -----------------------
def _lw_by_n(n_pisos: int, lw_min=0.65, lw_max=2.2):
    """Línea más fina a medida que crece n (30 pisos -> finito)."""
    n = max(int(n_pisos), 1)
    # 1 piso -> lw_max ; 30+ -> cerca de lw_min
    if n <= 3:
        return lw_max
    t = min(max((n - 3) / (30 - 3), 0.0), 1.0)
    return float(lw_max * (1 - t) + lw_min * t)

def _ms_by_n(n_pisos: int, ms_min=2.6, ms_max=5.5):
    """Marker size adaptativo."""
    n = max(int(n_pisos), 1)
    if n <= 3:
        return ms_max
    t = min(max((n - 3) / (30 - 3), 0.0), 1.0)
    return float(ms_max * (1 - t) + ms_min * t)

def _df_to_compact_table(df: pd.DataFrame, height_min=150, height_max=300):
    """Altura compacta (no se va gigante) según filas."""
    n = len(df)
    # aprox 26px por fila + header
    h = 40 + 26 * min(n, 8)
    h = int(max(height_min, min(height_max, h)))
    st.dataframe(df, hide_index=True, use_container_width=True, height=h)

def _plot_story_shear_etabs(V_story, y_levels, title, color_line):
    """
    Diagrama tipo ETABS (stairs) y simétrico (+V y -V), SIN cerrar en 0.

    Entradas:
      - V_story  : (n_pisos,)  cortante por piso (una grada por piso)
      - y_levels : (n_pisos+1,) alturas de niveles [0, h1, h2, ..., hn]
    """
    V_story  = np.asarray(V_story, float).ravel()
    y_levels = np.asarray(y_levels, float).ravel()

    if len(y_levels) != len(V_story) + 1:
        st.error(
            f"❌ _plot_story_shear_etabs: y_levels debe ser n+1 y V_story n. "
            f"V_story={V_story.shape}, y_levels={y_levels.shape}"
        )
        return
    if len(V_story) == 0:
        st.warning("⚠️ V_story vacío.")
        return

    # ordenar por altura (por si acaso)
    order = np.argsort(y_levels)
    y_levels = y_levels[order]

    n = len(V_story)
    lw = _lw_by_n(n)
    ms = _ms_by_n(n)

    # ------------------ construir "stairs" para +V ------------------
    xs_pos, ys = [], []

    # arranque en base
    xs_pos.append(V_story[0]); ys.append(y_levels[0])

    for i in range(n):
        # vertical del entrepiso i
        xs_pos += [V_story[i], V_story[i]]
        ys     += [y_levels[i], y_levels[i+1]]

        # horizontal en el nivel superior SOLO si hay siguiente piso
        if i < n - 1:
            xs_pos += [V_story[i+1]]
            ys     += [y_levels[i+1]]

    xs_pos = np.asarray(xs_pos, float)
    ys     = np.asarray(ys, float)
    xs_neg = -xs_pos

    # ------------------- plot -------------------
    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # NaN para romper la línea entre (-) y (+)
    X = np.r_[xs_neg, np.nan, xs_pos]
    Y = np.r_[ys,     np.nan, ys]
    ax.plot(X, Y, "-", color=color_line, lw=lw)

    # puntos en niveles (base + pisos)
    y_pts = y_levels
    x_pts_pos = np.r_[V_story[0], V_story]      # (n+1)
    x_pts_pos = x_pts_pos[:len(y_pts)]
    x_pts_neg = -x_pts_pos

    ax.plot(x_pts_pos, y_pts, "o", color=color_line, ms=ms)
    ax.plot(x_pts_neg, y_pts, "o", color=color_line, ms=ms, alpha=0.85)

    ax.axvline(0.0, color=COLOR_GRID, lw=1.0, alpha=0.6)

    ax.set_xlabel("Cortante V [Tf]", color=COLOR_TEXT)
    ax.set_ylabel("Altura [m]", color=COLOR_TEXT)
    ax.set_title(title, color=COLOR_TEXT, fontweight="bold")
    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
    ax.tick_params(colors=COLOR_TEXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    vmax = float(np.max(np.abs(V_story))) if len(V_story) else 1.0
    vmax = 1.0 if vmax <= 0 else vmax
    ax.set_xlim(-1.10 * vmax, 1.10 * vmax)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# -------------------- Selector método --------------------
metodo = st.selectbox(
    "Selecciona el método de cortantes",
    ["Análisis modal espectral (RSA)", "Tiempo historia (THA)"],
    index=0,
    key="metodo_cortantes"
)

# Alturas pisos (FIJA) -> de bloque Newmark se guardó como "alturas"
alt_fix = st.session_state.get("alturas", None)
if alt_fix is None:
    st.error("❌ Falta st.session_state['alturas'] (alturas de pisos).")
    st.stop()
alt_fix = np.asarray(alt_fix, float).ravel()

# Niveles para ambos (0 + alturas de pisos)
y_levels = np.r_[0.0, alt_fix]   # (n_pisos+1,)
n_pisos = len(alt_fix)

# =============================================================================
# RSA (Modal espectral) – FIJA vs AISLADA
# =============================================================================
if metodo == "Análisis modal espectral (RSA)":

    # ---- espectro NEC24 guardado ----
    T_spec        = st.session_state.get("T_spec", None)
    Sa_elast_spec = st.session_state.get("Sa_elast_spec", None)
    Sa_inel_spec  = st.session_state.get("Sa_inelas_spec", None)

    if T_spec is None or Sa_elast_spec is None or Sa_inel_spec is None:
        st.error("❌ Falta el espectro guardado: T_spec / Sa_elast_spec / Sa_inelas_spec.")
        st.stop()

    T_spec        = np.asarray(T_spec, float).ravel()
    Sa_elast_spec = np.asarray(Sa_elast_spec, float).ravel()
    Sa_inel_spec  = np.asarray(Sa_inel_spec, float).ravel()

    if not (len(T_spec) == len(Sa_elast_spec) == len(Sa_inel_spec)):
        st.error("❌ Espectro: dimensiones no coinciden.")
        st.stop()

    tipo_sa = st.selectbox("Sa a usar en RSA", ["Inelástico (R)", "Elástico"], index=0, key="sa_use_shear_rsa")
    Sa_use = Sa_inel_spec if "Inelástico" in tipo_sa else Sa_elast_spec

    # ---- variables modales FIJA ----
    M_fix  = st.session_state.get("M_cond", None)
    Vn_fix = st.session_state.get("v_norm_sin", None)
    T_fix  = st.session_state.get("T_sin", None)
    if M_fix is None or Vn_fix is None or T_fix is None:
        st.error("❌ Falta FIJA: M_cond / v_norm_sin / T_sin.")
        st.stop()

    M_fix  = np.asarray(M_fix, float)
    Vn_fix = np.asarray(Vn_fix, float)
    T_fix  = np.asarray(T_fix, float).ravel()
    n_fix  = M_fix.shape[0]

    # ---- variables modales AISLADA ----
    M_ais = st.session_state.get("M_cond_ais", None)
    if M_ais is None:
        M_ais = st.session_state.get("M_cond_aislador", None)
    Vn_ais = st.session_state.get("v_norm_ais", None)
    T_ais  = st.session_state.get("T_ais", None)
    if M_ais is None or Vn_ais is None or T_ais is None:
        st.error("❌ Falta AISLADA: M_cond_ais (o M_cond_aislador) / v_norm_ais / T_ais.")
        st.stop()

    M_ais  = np.asarray(M_ais, float)
    Vn_ais = np.asarray(Vn_ais, float)
    T_ais  = np.asarray(T_ais, float).ravel()
    n_ais  = M_ais.shape[0]

    if Vn_fix.shape[0] != n_fix:
        st.error("❌ FIJA: v_norm_sin no coincide con n_dofs.")
        st.stop()
    if Vn_ais.shape[0] != n_ais:
        st.error("❌ AISLADA: v_norm_ais no coincide con n_dofs.")
        st.stop()

    # ---- RSA: fuerzas modales por inercia y cortantes por nivel (SRSS) ----
    r_fix = np.ones((n_fix, 1))
    m_fix = np.diag(M_fix).reshape(n_fix, 1)

    V_fix_modes = []
    for r in range(Vn_fix.shape[1]):
        Tr = float(T_fix[r]) if r < len(T_fix) else np.nan
        if not np.isfinite(Tr) or Tr <= 0:
            continue
        Sa_r = float(np.interp(Tr, T_spec, Sa_use)) * 9.81
        phi  = Vn_fix[:, r].reshape(n_fix, 1)
        Gamma = float((phi.T @ M_fix @ r_fix).item())
        a_r   = Gamma * phi * Sa_r
        F_r   = m_fix * a_r

        V_r = np.zeros((n_fix, 1))
        for k in range(n_fix):
            V_r[k, 0] = np.sum(F_r[k:, 0])
        V_fix_modes.append(V_r[:, 0])

    if len(V_fix_modes) == 0:
        st.error("❌ RSA FIJA: no pude armar modos válidos.")
        st.stop()

    V_fix_srss = np.sqrt(np.sum(np.vstack(V_fix_modes) ** 2, axis=0))  # (n_fix,)

    # AISLADA
    r_ais = np.ones((n_ais, 1))
    m_ais = np.diag(M_ais).reshape(n_ais, 1)

    V_ais_modes = []
    for r in range(Vn_ais.shape[1]):
        Tr = float(T_ais[r]) if r < len(T_ais) else np.nan
        if not np.isfinite(Tr) or Tr <= 0:
            continue
        Sa_r = float(np.interp(Tr, T_spec, Sa_use)) * 9.81
        phi  = Vn_ais[:, r].reshape(n_ais, 1)
        Gamma = float((phi.T @ M_ais @ r_ais).item())
        a_r   = Gamma * phi * Sa_r
        F_r   = m_ais * a_r

        V_r = np.zeros((n_ais, 1))
        for k in range(n_ais):
            V_r[k, 0] = np.sum(F_r[k:, 0])
        V_ais_modes.append(V_r[:, 0])

    if len(V_ais_modes) == 0:
        st.error("❌ RSA AISLADA: no pude armar modos válidos.")
        st.stop()

    V_ais_srss = np.sqrt(np.sum(np.vstack(V_ais_modes) ** 2, axis=0))  # (n_ais,)

    # --------- Convertir a "cortantes por piso" comparables ---------
    if len(V_fix_srss) != n_pisos:
        st.error(f"❌ RSA FIJA: V={V_fix_srss.shape} no coincide con n_pisos={n_pisos}.")
        st.stop()
    V_fix_story = V_fix_srss.copy()

    if len(V_ais_srss) == n_pisos + 1:
        V_ais_story = V_ais_srss[:-1]      # sin el último para comparar piso a piso
    elif len(V_ais_srss) == n_pisos:
        V_ais_story = V_ais_srss.copy()
    else:
        st.error(f"❌ RSA AISLADA: V={V_ais_srss.shape} no coincide con n_pisos={n_pisos} ni n_pisos+1.")
        st.stop()

    # Tablas compactas (en expander)
    df_fix = pd.DataFrame({
        "Piso": np.arange(1, n_pisos + 1),
        "Altura sup [m]": np.round(alt_fix, 3),
        "V_rsa [Tf]": np.round(V_fix_story, 3),
    })
    df_ais = pd.DataFrame({
        "Piso": np.arange(1, n_pisos + 1),
        "Altura sup [m]": np.round(alt_fix, 3),
        "V_rsa [Tf]": np.round(V_ais_story, 3),
    })

    colL, colR = st.columns([1, 1], gap="large")

    with colL:
        with st.container(border=True):
            st.subheader("🟦 FIJA – RSA (SRSS)")

            with st.expander("📋 Ver tabla de cortantes (FIJA)", expanded=False):
                _df_to_compact_table(df_fix)

            _plot_story_shear_etabs(V_fix_story, y_levels, "Cortantes por piso – RSA (FIJA)", COLOR_FIX)

    with colR:
        with st.container(border=True):
            st.subheader("🟩 AISLADA – RSA (SRSS)")

            with st.expander("📋 Ver tabla de cortantes (AISLADA)", expanded=False):
                _df_to_compact_table(df_ais)

            _plot_story_shear_etabs(V_ais_story, y_levels, "Cortantes por piso – RSA (AISLADA)", COLOR_AIS)

    st.success("✅ RSA listo (SRSS) – FIJA vs AISLADA (mismas gradas por piso).")

    # Guardar para comparativo FINAL
    st.session_state["cmp_V_fix_story"] = np.asarray(V_fix_story, float).ravel()
    st.session_state["cmp_V_ais_story"] = np.asarray(V_ais_story, float).ravel()
    st.session_state["cmp_tag_shear"]   = "RSA (SRSS)"

# =============================================================================
# THA (Tiempo historia) – FIJA vs AISLADA
# =============================================================================
else:
    modo_tha = st.selectbox("Modo THA", ["Tiempo", "Máximos–mínimos (absolutos)"], index=0)

    dt = st.session_state.get("dt", None)
    ag = st.session_state.get("ag_filt", None)

    a_fix = st.session_state.get("a_t", None)
    Mc    = st.session_state.get("M_cond", None)

    a_ais = st.session_state.get("a_t_ais", None)
    M_ais = st.session_state.get("M_cond_ais", None)
    if M_ais is None:
        M_ais = st.session_state.get("M_cond_aislador", None)

    falt = []
    if dt is None: falt.append("dt")
    if ag is None: falt.append("ag_filt")
    if a_fix is None: falt.append("a_t")
    if Mc is None: falt.append("M_cond")
    if a_ais is None: falt.append("a_t_ais")
    if M_ais is None: falt.append("M_cond_ais (o M_cond_aislador)")
    if falt:
        st.error("❌ Faltan variables THA: " + ", ".join(falt))
        st.stop()

    dt = float(dt)
    ag = np.asarray(ag, float).ravel()

    a_fix = np.asarray(a_fix, float);  a_fix = a_fix if a_fix.ndim == 2 else a_fix[np.newaxis, :]
    a_ais = np.asarray(a_ais, float);  a_ais = a_ais if a_ais.ndim == 2 else a_ais[np.newaxis, :]

    def _match_ag(ag_in, nt):
        ag2 = np.asarray(ag_in, float).ravel()
        if len(ag2) < nt:
            ag2 = np.pad(ag2, (0, nt - len(ag2)), mode="constant")
        else:
            ag2 = ag2[:nt]
        t = np.arange(nt, dtype=float) * dt
        return t, ag2

    def _story_shears(Mmat, a_rel, ag_series):
        a_rel = np.asarray(a_rel, float)
        n, nt = a_rel.shape
        m = np.diag(np.asarray(Mmat, float)).reshape(n, 1)
        a_abs = a_rel + ag_series.reshape(1, nt)
        F = m * a_abs
        V = np.zeros_like(F)
        for k in range(n):
            V[k, :] = np.sum(F[k:, :], axis=0)
        return V

    t_fix, ag_fix = _match_ag(ag, a_fix.shape[1])
    V_fix = _story_shears(Mc, a_fix, ag_fix)          # (n_pisos, nt)

    t_ais, ag_ais = _match_ag(ag, a_ais.shape[1])
    V_ais = _story_shears(M_ais, a_ais, ag_ais)        # (n_pisos+1?, nt)

    if V_fix.shape[0] != n_pisos:
        st.error("❌ THA FIJA: V_fix no calza con n_pisos.")
        st.stop()

    # Para comparar: AISLADA -> tomar SOLO n_pisos
    if V_ais.shape[0] == n_pisos + 1:
        V_ais_use = V_ais[:-1, :]
    elif V_ais.shape[0] == n_pisos:
        V_ais_use = V_ais
    else:
        st.error("❌ THA AISLADA: V_ais no calza con n_pisos ni n_pisos+1.")
        st.stop()

    # Selector de tiempo
    if modo_tha == "Tiempo":
        tmax = float(min(t_fix[-1], t_ais[-1]))
        t_sel = st.slider("Selecciona el tiempo t [s]", 0.0, tmax, min(5.0, tmax), 0.01)
        i_fix = int(np.argmin(np.abs(t_fix - t_sel)))
        i_ais = int(np.argmin(np.abs(t_ais - t_sel)))

    # Vector final a guardar
    if modo_tha == "Tiempo":
        V_fix_story = V_fix[:, i_fix]
        V_ais_story = V_ais_use[:, i_ais]
    else:
        V_fix_story = np.max(np.abs(V_fix), axis=1)
        V_ais_story = np.max(np.abs(V_ais_use), axis=1)

    colL, colR = st.columns([1, 1], gap="large")

    with colL:
        with st.container(border=True):
            st.subheader("🟦 FIJA – THA")

            if modo_tha == "Tiempo":
                Vp = V_fix[:, i_fix]
                st.caption(f"t ≈ {t_fix[i_fix]:.3f} s")

                df = pd.DataFrame({
                    "Piso": np.arange(1, n_pisos + 1),
                    "Altura sup [m]": np.round(alt_fix, 3),
                    "V(t) [Tf]": np.round(Vp, 3),
                })
                with st.expander("📋 Ver tabla de cortantes (FIJA)", expanded=False):
                    _df_to_compact_table(df)

                _plot_story_shear_etabs(Vp, y_levels, "THA – Cortantes por piso (FIJA)", COLOR_FIX)

            else:
                Vabs = np.max(np.abs(V_fix), axis=1)

                df = pd.DataFrame({
                    "Piso": np.arange(1, n_pisos + 1),
                    "Altura sup [m]": np.round(alt_fix, 3),
                    "Vabs_max [Tf]": np.round(Vabs, 3),
                })
                with st.expander("📋 Ver tabla de cortantes (FIJA)", expanded=False):
                    _df_to_compact_table(df)

                _plot_story_shear_etabs(Vabs, y_levels, "THA – Cortante absoluto máximo (FIJA)", COLOR_GUIDE)

    with colR:
        with st.container(border=True):
            st.subheader("🟩 AISLADA – THA")

            if modo_tha == "Tiempo":
                Vp = V_ais_use[:, i_ais]
                st.caption(f"t ≈ {t_ais[i_ais]:.3f} s")

                df = pd.DataFrame({
                    "Piso": np.arange(1, n_pisos + 1),
                    "Altura sup [m]": np.round(alt_fix, 3),
                    "V(t) [Tf]": np.round(Vp, 3),
                })
                with st.expander("📋 Ver tabla de cortantes (AISLADA)", expanded=False):
                    _df_to_compact_table(df)

                _plot_story_shear_etabs(Vp, y_levels, "THA – Cortantes por piso (AISLADA)", COLOR_AIS)

            else:
                Vabs = np.max(np.abs(V_ais_use), axis=1)

                df = pd.DataFrame({
                    "Piso": np.arange(1, n_pisos + 1),
                    "Altura sup [m]": np.round(alt_fix, 3),
                    "Vabs_max [Tf]": np.round(Vabs, 3),
                })
                with st.expander("📋 Ver tabla de cortantes (AISLADA)", expanded=False):
                    _df_to_compact_table(df)

                _plot_story_shear_etabs(Vabs, y_levels, "THA – Cortante absoluto máximo (AISLADA)", COLOR_GUIDE)

    st.success("✅ THA listo – FIJA vs AISLADA (mismas gradas por piso).")

    # Guardar para comparativo FINAL
    st.session_state["cmp_V_fix_story"] = np.asarray(V_fix_story, float).ravel()
    st.session_state["cmp_V_ais_story"] = np.asarray(V_ais_story, float).ravel()
    st.session_state["cmp_tag_shear"]   = "THA (Tiempo)" if modo_tha == "Tiempo" else "THA (Máx abs)"

# =============================================================================
# === BLOQUE 9: DESPLAZAMIENTOS LATERALES (RSA vs THA) – FIJA vs AISLADA =======
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

st.markdown("## 🟦 Desplazamientos laterales – Modal espectral (RSA) vs Tiempo historia (THA)")

# ----------------------- Estilos -----------------------
BG            = "#2B3141"
COLOR_TEXT    = "#E8EDF2"
COLOR_GRID    = "#5B657A"
COLOR_FIX     = "#FFE6A3"
COLOR_AIS     = "#77DD77"
COLOR_GUIDE   = "#FFDFA0"
HALO = [pe.withStroke(linewidth=2.4, foreground=BG), pe.Normal()]

# ----------------------- Helpers (adaptativo) -----------------------
def _lw_by_n(n_pisos: int, lw_min=0.65, lw_max=2.0):
    n = max(int(n_pisos), 1)
    if n <= 3:
        return lw_max
    t = min(max((n - 3) / (30 - 3), 0.0), 1.0)
    return float(lw_max * (1 - t) + lw_min * t)

def _ms_by_n(n_pisos: int, ms_min=2.6, ms_max=5.5):
    n = max(int(n_pisos), 1)
    if n <= 3:
        return ms_max
    t = min(max((n - 3) / (30 - 3), 0.0), 1.0)
    return float(ms_max * (1 - t) + ms_min * t)

def _df_to_compact_table(df: pd.DataFrame, height_min=150, height_max=300):
    n = len(df)
    h = 40 + 26 * min(n, 8)
    h = int(max(height_min, min(height_max, h)))
    st.dataframe(df, hide_index=True, use_container_width=True, height=h)

def _plot_profile(U, y, title, color_line, n_ref=10, xlabel="Desplazamiento u [m]"):
    U = np.asarray(U, float).ravel()
    y = np.asarray(y, float).ravel()

    n_pisos = max(len(y) - 1, 1)  # para escalar estilo (base incluida)
    lw = _lw_by_n(n_pisos)
    ms = _ms_by_n(n_pisos)

    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(U, y, "-o", color=color_line, lw=lw, ms=ms)

    ax.axvline(0.0, color=COLOR_GRID, lw=1.0, alpha=0.6)

    ax.set_xlabel(xlabel, color=COLOR_TEXT)
    ax.set_ylabel("Altura [m]", color=COLOR_TEXT)
    ax.set_title(title, color=COLOR_TEXT, fontweight="bold")
    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
    ax.tick_params(colors=COLOR_TEXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# -------------------- Selector método --------------------
metodo = st.selectbox(
    "Selecciona el método de desplazamientos",
    ["Análisis modal espectral (RSA)", "Tiempo historia (THA)"],
    index=0,
    key="metodo_desplazamientos"
)

# Alturas pisos (FIJA)
alt_fix = st.session_state.get("alturas", None)
if alt_fix is None:
    st.error("❌ Falta st.session_state['alturas'] (alturas de pisos).")
    st.stop()
alt_fix = np.asarray(alt_fix, float).ravel()
n_pisos = len(alt_fix)

# =============================================================================
# RSA (Modal espectral) – FIJA vs AISLADA
# =============================================================================
if metodo == "Análisis modal espectral (RSA)":

    # ---- espectro NEC24 guardado ----
    T_spec        = st.session_state.get("T_spec", None)
    Sa_elast_spec = st.session_state.get("Sa_elast_spec", None)
    Sa_inel_spec  = st.session_state.get("Sa_inelas_spec", None)

    if T_spec is None or Sa_elast_spec is None or Sa_inel_spec is None:
        st.error("❌ Falta el espectro guardado: T_spec / Sa_elast_spec / Sa_inelas_spec.")
        st.stop()

    T_spec        = np.asarray(T_spec, float).ravel()
    Sa_elast_spec = np.asarray(Sa_elast_spec, float).ravel()
    Sa_inel_spec  = np.asarray(Sa_inel_spec, float).ravel()

    if not (len(T_spec) == len(Sa_elast_spec) == len(Sa_inel_spec)):
        st.error(
            f"❌ Espectro: dimensiones no coinciden: "
            f"T={T_spec.shape}, Sa_elast={Sa_elast_spec.shape}, Sa_inel={Sa_inel_spec.shape}"
        )
        st.stop()

    tipo_sa = st.selectbox("Sa a usar en RSA", ["Inelástico (R)", "Elástico"], index=0, key="sa_use_disp_rsa")
    Sa_use = Sa_inel_spec if "Inelástico" in tipo_sa else Sa_elast_spec

    # ---- variables modales FIJA ----
    M_fix  = st.session_state.get("M_cond", None)
    Vn_fix = st.session_state.get("v_norm_sin", None)
    T_fix  = st.session_state.get("T_sin", None)
    w_fix  = st.session_state.get("w_sin", None)
    if M_fix is None or Vn_fix is None or T_fix is None:
        st.error("❌ Falta FIJA: M_cond / v_norm_sin / T_sin.")
        st.stop()

    M_fix  = np.asarray(M_fix, float)
    Vn_fix = np.asarray(Vn_fix, float)
    T_fix  = np.asarray(T_fix, float).ravel()
    n_fix  = M_fix.shape[0]

    # ---- variables modales AISLADA ----
    M_ais = st.session_state.get("M_cond_ais", None)
    if M_ais is None:
        M_ais = st.session_state.get("M_cond_aislador", None)
    Vn_ais = st.session_state.get("v_norm_ais", None)
    T_ais  = st.session_state.get("T_ais", None)
    w_ais  = st.session_state.get("w_ais", None)
    if M_ais is None or Vn_ais is None or T_ais is None:
        st.error("❌ Falta AISLADA: M_cond_ais (o M_cond_aislador) / v_norm_ais / T_ais.")
        st.stop()

    M_ais  = np.asarray(M_ais, float)
    Vn_ais = np.asarray(Vn_ais, float)
    T_ais  = np.asarray(T_ais, float).ravel()
    n_ais  = M_ais.shape[0]

    # ---- chequeos dims ----
    if Vn_fix.shape[0] != n_fix:
        st.error(f"❌ FIJA: v_norm_sin filas {Vn_fix.shape[0]} != n_dofs {n_fix}")
        st.stop()
    if Vn_ais.shape[0] != n_ais:
        st.error(f"❌ AISLADA: v_norm_ais filas {Vn_ais.shape[0]} != n_dofs {n_ais}")
        st.stop()

    # ---- RSA: desplazamientos (SRSS) ----
    g = 9.81

    r_fix = np.ones((n_fix, 1), float)
    U_fix_modes = []
    for r in range(Vn_fix.shape[1]):
        Tr = float(T_fix[r]) if r < len(T_fix) else np.nan
        if not np.isfinite(Tr) or Tr <= 0:
            continue
        Sa_r_g = float(np.interp(Tr, T_spec, Sa_use))
        Sa_r   = Sa_r_g * g
        phi = Vn_fix[:, r].reshape(n_fix, 1)
        Gamma = float((phi.T @ M_fix @ r_fix).item())

        if w_fix is not None and r < len(np.asarray(w_fix, float).ravel()):
            wv = float(np.asarray(w_fix, float).ravel()[r])
        else:
            wv = 2.0 * np.pi / Tr

        if not np.isfinite(wv) or wv <= 0:
            continue

        q_r = Gamma * Sa_r / (wv**2)
        U_fix_modes.append((phi * q_r).ravel())

    if len(U_fix_modes) == 0:
        st.error("❌ RSA FIJA: no pude armar modos válidos (revisa T_sin / w_sin / v_norm_sin).")
        st.stop()

    U_fix_modes = np.vstack(U_fix_modes)
    u_fix_srss  = np.sqrt(np.sum(U_fix_modes**2, axis=0))
    st.session_state["u_fix_srss"] = u_fix_srss

    r_ais = np.ones((n_ais, 1), float)
    U_ais_modes = []
    for r in range(Vn_ais.shape[1]):
        Tr = float(T_ais[r]) if r < len(T_ais) else np.nan
        if not np.isfinite(Tr) or Tr <= 0:
            continue
        Sa_r_g = float(np.interp(Tr, T_spec, Sa_use))
        Sa_r   = Sa_r_g * g
        phi = Vn_ais[:, r].reshape(n_ais, 1)
        Gamma = float((phi.T @ M_ais @ r_ais).item())

        if w_ais is not None and r < len(np.asarray(w_ais, float).ravel()):
            wv = float(np.asarray(w_ais, float).ravel()[r])
        else:
            wv = 2.0 * np.pi / Tr

        if not np.isfinite(wv) or wv <= 0:
            continue

        q_r = Gamma * Sa_r / (wv**2)
        U_ais_modes.append((phi * q_r).ravel())

    if len(U_ais_modes) == 0:
        st.error("❌ RSA AISLADA: no pude armar modos válidos (revisa T_ais / w_ais / v_norm_ais).")
        st.stop()

    U_ais_modes = np.vstack(U_ais_modes)
    u_ais_srss  = np.sqrt(np.sum(U_ais_modes**2, axis=0))
    st.session_state["u_ais_srss"] = u_ais_srss

    # Ejes de altura para AISLADA: [0 (aislador), pisos...]
    y_ais = np.concatenate([[0.0], alt_fix])

    # Validación tamaños
    if len(u_fix_srss) != len(alt_fix):
        st.error(f"❌ RSA FIJA: u={u_fix_srss.shape} no coincide con alturas={alt_fix.shape}.")
        st.stop()
    if len(u_ais_srss) != len(y_ais):
        st.error(f"❌ RSA AISLADA: u={u_ais_srss.shape} no coincide con y_ais={y_ais.shape}.")
        st.stop()

    # Plot FIJA incluye base 0
    y_fix_plot = np.concatenate([[0.0], alt_fix])
    u_fix_plot = np.concatenate([[0.0], u_fix_srss])

    # Tablas
    dfL = pd.DataFrame({
        "Piso": np.arange(1, len(u_fix_srss) + 1),
        "Altura [m]": np.round(alt_fix, 3),
        "u_rsa [m]": np.round(u_fix_srss, 6),
        "u_rsa [mm]": np.round(u_fix_srss * 1000.0, 3),
    })

    dfR = pd.DataFrame({
        "Nivel": ["Aislador"] + [f"Piso {i}" for i in range(1, len(u_ais_srss))],
        "Altura [m]": np.round(y_ais, 3),
        "u_rsa [m]": np.round(u_ais_srss, 6),
        "u_rsa [mm]": np.round(u_ais_srss * 1000.0, 3),
    })

    colL, colR = st.columns([1, 1], gap="large")

    with colL:
        with st.container(border=True):
            st.subheader("🟦 FIJA – RSA (SRSS)")
            with st.expander("📋 Ver tabla de desplazamientos (FIJA)", expanded=False):
                _df_to_compact_table(dfL)
            _plot_profile(u_fix_plot, y_fix_plot, "Desplazamientos por piso – RSA (FIJA)", COLOR_FIX)

    with colR:
        with st.container(border=True):
            st.subheader("🟩 AISLADA – RSA (SRSS, incluye aislador)")
            with st.expander("📋 Ver tabla de desplazamientos (AISLADA)", expanded=False):
                _df_to_compact_table(dfR)
            _plot_profile(u_ais_srss, y_ais, "Desplazamientos por nivel – RSA (AISLADA)", COLOR_AIS)

    st.success("✅ RSA listo (SRSS) – FIJA vs AISLADA.")

    # Guardar para comparativo FINAL
    st.session_state["cmp_U_fix_levels"] = np.asarray(np.r_[0.0, u_fix_srss], float).ravel()  # base + pisos
    st.session_state["cmp_U_ais_levels"] = np.asarray(u_ais_srss, float).ravel()              # aislador + pisos
    st.session_state["cmp_tag_disp"]     = "RSA (SRSS)"

# =============================================================================
# THA (Tiempo historia) – FIJA vs AISLADA
# =============================================================================
else:
    modo_tha = st.selectbox("Modo THA", ["Tiempo", "Máximos–mínimos (absolutos)"], index=0, key="modo_tha_disp")

    dt = st.session_state.get("dt", None)
    u_fix = st.session_state.get("u_t", None)      # (n_pisos, nt)
    u_ais = st.session_state.get("u_t_ais", None)  # (n_pisos+1, nt) incluye aislador

    falt = []
    if dt is None: falt.append("dt")
    if u_fix is None: falt.append("u_t")
    if u_ais is None: falt.append("u_t_ais")
    if falt:
        st.error("❌ Faltan variables THA: " + ", ".join(falt))
        st.stop()

    dt = float(dt)
    u_fix = np.asarray(u_fix, float);  u_fix = u_fix if u_fix.ndim == 2 else u_fix[np.newaxis, :]
    u_ais = np.asarray(u_ais, float);  u_ais = u_ais if u_ais.ndim == 2 else u_ais[np.newaxis, :]

    y_ais = np.concatenate([[0.0], alt_fix])

    if u_fix.shape[0] != len(alt_fix):
        st.error(f"❌ THA FIJA: u_fix={u_fix.shape} no calza con alturas={alt_fix.shape}.")
        st.stop()
    if u_ais.shape[0] != len(y_ais):
        st.error(f"❌ THA AISLADA: u_ais={u_ais.shape} no calza con y_ais={y_ais.shape}.")
        st.stop()

    t_fix = np.arange(u_fix.shape[1], dtype=float) * dt
    t_ais = np.arange(u_ais.shape[1], dtype=float) * dt

    if modo_tha == "Tiempo":
        tmax = float(min(t_fix[-1], t_ais[-1]))
        t_sel = st.slider("Selecciona el tiempo t [s]", 0.0, tmax, min(5.0, tmax), 0.01, key="t_sel_disp")
        i_fix = int(np.argmin(np.abs(t_fix - t_sel)))
        i_ais = int(np.argmin(np.abs(t_ais - t_sel)))

        U_fix_out = u_fix[:, i_fix]
        U_ais_out = u_ais[:, i_ais]
        tag = f"t ≈ {t_fix[i_fix]:.3f} s"
    else:
        U_fix_out = np.max(np.abs(u_fix), axis=1)
        U_ais_out = np.max(np.abs(u_ais), axis=1)
        tag = "máximos absolutos"

    # Guardar para comparativo FINAL
    st.session_state["cmp_U_fix_levels"] = np.asarray(np.r_[0.0, U_fix_out], float).ravel()
    st.session_state["cmp_U_ais_levels"] = np.asarray(U_ais_out, float).ravel()
    st.session_state["cmp_tag_disp"]     = f"THA ({tag})"

    # Tablas
    dfL = pd.DataFrame({
        "Piso": np.arange(1, len(U_fix_out) + 1),
        "Altura [m]": np.round(alt_fix, 3),
        "u [m]": np.round(U_fix_out, 6),
        "u [mm]": np.round(U_fix_out * 1000.0, 3),
    })

    dfR = pd.DataFrame({
        "Nivel": ["Aislador"] + [f"Piso {i}" for i in range(1, len(U_ais_out))],
        "Altura [m]": np.round(y_ais, 3),
        "u [m]": np.round(U_ais_out, 6),
        "u [mm]": np.round(U_ais_out * 1000.0, 3),
    })

    colL, colR = st.columns([1, 1], gap="large")

    with colL:
        with st.container(border=True):
            st.subheader(f"🟦 FIJA – THA ({tag})")
            with st.expander("📋 Ver tabla de desplazamientos (FIJA)", expanded=False):
                _df_to_compact_table(dfL)
            _plot_profile(np.r_[0.0, U_fix_out], np.r_[0.0, alt_fix], "THA – Perfil de desplazamientos (FIJA)", COLOR_FIX)

    with colR:
        with st.container(border=True):
            st.subheader(f"🟩 AISLADA – THA ({tag})")
            with st.expander("📋 Ver tabla de desplazamientos (AISLADA)", expanded=False):
                _df_to_compact_table(dfR)
            _plot_profile(U_ais_out, y_ais, "THA – Perfil de desplazamientos (AISLADA)", COLOR_AIS)

    st.success("✅ THA listo – FIJA vs AISLADA (incluye desplazamiento del aislador).")

# =============================================================================
# === BLOQUE 10: DERIVAS POR ENTREPISO (CONSISTENTE CON DESPLAZAMIENTOS) ======
# === Fuente: HEREDA lo seleccionado en DESPLAZAMIENTOS (RSA o THA) ===========
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

st.markdown("## 📐 Derivas por entrepiso")

# ----------------------- Estilos -----------------------
BG            = "#2B3141"
COLOR_TEXT    = "#E8EDF2"
COLOR_GRID    = "#5B657A"
COLOR_FIX     = "#FFE6A3"
COLOR_AIS     = "#77DD77"
COLOR_GUIDE   = "#FFDFA0"
HALO = [pe.withStroke(linewidth=2.4, foreground=BG), pe.Normal()]

# ----------------------- Helpers (adaptativo) -----------------------
def _lw_by_n(n_pisos: int, lw_min=0.65, lw_max=2.0):
    n = max(int(n_pisos), 1)
    if n <= 3:
        return lw_max
    t = min(max((n - 3) / (30 - 3), 0.0), 1.0)
    return float(lw_max * (1 - t) + lw_min * t)

def _ms_by_n(n_pisos: int, ms_min=2.6, ms_max=5.5):
    n = max(int(n_pisos), 1)
    if n <= 3:
        return ms_max
    t = min(max((n - 3) / (30 - 3), 0.0), 1.0)
    return float(ms_max * (1 - t) + ms_min * t)

def _df_to_compact_table(df: pd.DataFrame, height_min=150, height_max=300):
    n = len(df)
    h = 40 + 26 * min(n, 8)
    h = int(max(height_min, min(height_max, h)))
    st.dataframe(df, hide_index=True, use_container_width=True, height=h)

def _calc_drifts(u_levels, y_levels):
    """
    u_levels: desplazamientos en niveles (incluye base/aislador)
    y_levels: alturas en niveles (incluye 0)
    Retorna:
      drift_i = (u_i - u_{i-1}) / (y_i - y_{i-1})   para i=1..n
      y_story = y_i (altura del nivel superior del entrepiso)
    """
    u_levels = np.asarray(u_levels, float).ravel()
    y_levels = np.asarray(y_levels, float).ravel()

    if len(u_levels) != len(y_levels):
        raise ValueError(f"u_levels y y_levels no coinciden: {u_levels.shape} vs {y_levels.shape}")
    if len(u_levels) < 2:
        raise ValueError("Se requieren al menos 2 niveles para calcular derivas.")

    dy = np.diff(y_levels)
    du = np.diff(u_levels)

    if np.any(np.abs(dy) < 1e-12):
        raise ValueError("Hay alturas repetidas (dy=0). Revisa tu vector de alturas.")

    drift = du / dy
    y_story = y_levels[1:]  # asociado al nivel superior del entrepiso
    return drift, y_story

def _plot_drift_sym(drift, y_story, title, color_line, xlabel="Deriva Δ/h", n_pisos_ref=10):
    """
    Plot simétrico (+drift y -drift) tipo ETABS.
    (línea + marcadores se ajustan según número de pisos)
    """
    drift = np.asarray(drift, float).ravel()
    y_story = np.asarray(y_story, float).ravel()

    n_pisos = max(int(n_pisos_ref), 1)
    lw = _lw_by_n(n_pisos)
    ms = _ms_by_n(n_pisos)

    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot( drift, y_story, "-o", color=color_line, lw=lw, ms=ms)
    ax.plot(-drift, y_story, "-o", color=color_line, lw=lw, ms=ms, alpha=0.82)
    ax.axvline(0.0, color=COLOR_GRID, lw=1.0, alpha=0.65)

    ax.set_xlabel(xlabel, color=COLOR_TEXT)
    ax.set_ylabel("Altura [m]", color=COLOR_TEXT)
    ax.set_title(title, color=COLOR_TEXT, fontweight="bold")

    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
    ax.tick_params(colors=COLOR_TEXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    vmax = float(np.max(np.abs(drift))) if len(drift) else 1.0
    vmax = 1.0 if vmax <= 0 else vmax
    ax.set_xlim(-1.10 * vmax, 1.10 * vmax)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Selector de tipo de deriva (REAL vs NEC24)
# -----------------------------------------------------------------------------
tipo_deriva = st.radio(
    "Tipo de deriva",
    ["Deriva real", "Deriva NEC24 (Cd·Δ / I)"],
    index=0,
    horizontal=True,
    key="tipo_deriva_out"
)

# -------------------- HEREDAR desde DESPLAZAMIENTOS --------------------
metodo_disp    = st.session_state.get("metodo_desplazamientos", None)
tipo_sa_disp   = st.session_state.get("sa_use_disp_rsa", None)  # solo aplica si RSA
modo_tha_disp  = st.session_state.get("modo_tha_disp", None)    # solo aplica si THA

if metodo_disp is None:
    st.error("❌ No encuentro 'metodo_desplazamientos' en session_state. Corre primero el bloque de DESPLAZAMIENTOS.")
    st.stop()

st.caption(f"🔗 Fuente: **{metodo_disp}** (heredado)")
if "RSA" in str(metodo_disp) and tipo_sa_disp is not None:
    st.caption(f"✅ Sa RSA: **{tipo_sa_disp}** (heredado)")

# -------------------- Alturas pisos (FIJA) --------------------
alt_fix = st.session_state.get("alturas", None)
if alt_fix is None:
    st.error("❌ Falta st.session_state['alturas'] (alturas de pisos).")
    st.stop()
alt_fix = np.asarray(alt_fix, float).ravel()
n_pisos = len(alt_fix)

# Niveles (arrancan en 0)
y_fix = np.r_[0.0, alt_fix]   # base + pisos
y_ais = np.r_[0.0, alt_fix]   # aislador (nivel 0) + pisos

# -----------------------------------------------------------------------------
# NEC24: pedir Cd e Ie (I) SOLO si el usuario pide deriva NEC
# -----------------------------------------------------------------------------
Cd = None
Ie = None
if "NEC24" in tipo_deriva:
    Cd = st.number_input(
        "Cd (NEC24)",
        min_value=0.1, max_value=20.0, value=5.5, step=0.1,
        key="nec_Cd_derivas"
    )

    Ie = st.session_state.get("nec_Ie", None)
    if Ie is None:
        nec_params = st.session_state.get("nec24_params", {})
        Ie = nec_params.get("Ie", None)

    if Ie is None:
        st.error("❌ No encuentro Ie (I) en session_state. Necesito st.session_state['nec_Ie'] o nec24_params['Ie'].")
        st.stop()

    Ie = float(Ie)
    if Ie <= 0:
        st.error("❌ Ie (I) inválido (<=0).")
        st.stop()

    st.caption(f"✅ Deriva_NEC = (Cd·Deriva_real)/I con Cd={float(Cd):.3g} e I={float(Ie):.3g}")

# =============================================================================
# ARMAR DESPLAZAMIENTOS DE SALIDA (u_levels) SEGÚN LA FUENTE HEREDADA
# =============================================================================
tag = ""

if "RSA" in str(metodo_disp):
    # --- Recalcular u_rsa (SRSS) aquí mismo (independiente) ---
    T_spec        = st.session_state.get("T_spec", None)
    Sa_elast_spec = st.session_state.get("Sa_elast_spec", None)
    Sa_inel_spec  = st.session_state.get("Sa_inelas_spec", None)

    if T_spec is None or Sa_elast_spec is None or Sa_inel_spec is None:
        st.error("❌ Falta el espectro guardado: T_spec / Sa_elast_spec / Sa_inelas_spec.")
        st.stop()

    T_spec        = np.asarray(T_spec, float).ravel()
    Sa_elast_spec = np.asarray(Sa_elast_spec, float).ravel()
    Sa_inel_spec  = np.asarray(Sa_inel_spec, float).ravel()

    if not (len(T_spec) == len(Sa_elast_spec) == len(Sa_inel_spec)):
        st.error("❌ Espectro: dimensiones no coinciden.")
        st.stop()

    if tipo_sa_disp is None:
        st.error("❌ No encuentro 'sa_use_disp_rsa' en session_state. Selecciona primero el tipo de Sa en DESPLAZAMIENTOS (RSA).")
        st.stop()

    Sa_use = Sa_inel_spec if "Inelástico" in str(tipo_sa_disp) else Sa_elast_spec

    # FIJA
    M_fix  = st.session_state.get("M_cond", None)
    Vn_fix = st.session_state.get("v_norm_sin", None)
    T_fix  = st.session_state.get("T_sin", None)
    w_fix  = st.session_state.get("w_sin", None)

    # AISLADA
    M_ais  = st.session_state.get("M_cond_ais", None)
    if M_ais is None:
        M_ais = st.session_state.get("M_cond_aislador", None)
    Vn_ais = st.session_state.get("v_norm_ais", None)
    T_ais  = st.session_state.get("T_ais", None)
    w_ais  = st.session_state.get("w_ais", None)

    if M_fix is None or Vn_fix is None or T_fix is None:
        st.error("❌ Falta FIJA: M_cond / v_norm_sin / T_sin.")
        st.stop()
    if M_ais is None or Vn_ais is None or T_ais is None:
        st.error("❌ Falta AISLADA: M_cond_ais (o M_cond_aislador) / v_norm_ais / T_ais.")
        st.stop()

    M_fix  = np.asarray(M_fix, float)
    Vn_fix = np.asarray(Vn_fix, float)
    T_fix  = np.asarray(T_fix, float).ravel()
    n_fix  = M_fix.shape[0]

    M_ais  = np.asarray(M_ais, float)
    Vn_ais = np.asarray(Vn_ais, float)
    T_ais  = np.asarray(T_ais, float).ravel()
    n_ais  = M_ais.shape[0]

    if Vn_fix.shape[0] != n_fix:
        st.error(f"❌ FIJA: v_norm_sin filas {Vn_fix.shape[0]} != n_dofs {n_fix}")
        st.stop()
    if Vn_ais.shape[0] != n_ais:
        st.error(f"❌ AISLADA: v_norm_ais filas {Vn_ais.shape[0]} != n_dofs {n_ais}")
        st.stop()

    g = 9.81

    # --- u_fix_srss ---
    r_fix = np.ones((n_fix, 1), float)
    U_fix_modes = []
    for r in range(Vn_fix.shape[1]):
        Tr = float(T_fix[r]) if r < len(T_fix) else np.nan
        if not np.isfinite(Tr) or Tr <= 0:
            continue

        Sa_r = float(np.interp(Tr, T_spec, Sa_use)) * g
        phi  = Vn_fix[:, r].reshape(n_fix, 1)
        Gamma = float((phi.T @ M_fix @ r_fix).item())

        if w_fix is not None and r < len(np.asarray(w_fix, float).ravel()):
            wv = float(np.asarray(w_fix, float).ravel()[r])
        else:
            wv = 2.0 * np.pi / Tr

        if not np.isfinite(wv) or wv <= 0:
            continue

        q_r = Gamma * Sa_r / (wv**2)
        U_fix_modes.append((phi * q_r).ravel())

    if len(U_fix_modes) == 0:
        st.error("❌ RSA FIJA: no pude armar modos válidos (revisa T_sin / w_sin / v_norm_sin).")
        st.stop()

    u_fix_srss = np.sqrt(np.sum(np.vstack(U_fix_modes)**2, axis=0))

    # --- u_ais_srss ---
    r_ais = np.ones((n_ais, 1), float)
    U_ais_modes = []
    for r in range(Vn_ais.shape[1]):
        Tr = float(T_ais[r]) if r < len(T_ais) else np.nan
        if not np.isfinite(Tr) or Tr <= 0:
            continue

        Sa_r = float(np.interp(Tr, T_spec, Sa_use)) * g
        phi  = Vn_ais[:, r].reshape(n_ais, 1)
        Gamma = float((phi.T @ M_ais @ r_ais).item())

        if w_ais is not None and r < len(np.asarray(w_ais, float).ravel()):
            wv = float(np.asarray(w_ais, float).ravel()[r])
        else:
            wv = 2.0 * np.pi / Tr

        if not np.isfinite(wv) or wv <= 0:
            continue

        q_r = Gamma * Sa_r / (wv**2)
        U_ais_modes.append((phi * q_r).ravel())

    if len(U_ais_modes) == 0:
        st.error("❌ RSA AISLADA: no pude armar modos válidos (revisa T_ais / w_ais / v_norm_ais).")
        st.stop()

    u_ais_srss = np.sqrt(np.sum(np.vstack(U_ais_modes)**2, axis=0))

    # Validación tamaños
    y_ais_check = np.r_[0.0, alt_fix]
    if len(u_fix_srss) != len(alt_fix):
        st.error("❌ RSA FIJA: tamaño u no coincide con alturas.")
        st.stop()
    if len(u_ais_srss) != len(y_ais_check):
        st.error("❌ RSA AISLADA: tamaño u no coincide con y_ais.")
        st.stop()

    u_fix_levels = np.r_[0.0, u_fix_srss]  # base=0
    u_ais_levels = u_ais_srss              # incluye aislador=0
    tag = "RSA (SRSS)"

else:
    # THA heredado
    dt = st.session_state.get("dt", None)
    u_fix = st.session_state.get("u_t", None)      # (n_pisos, nt)
    u_ais = st.session_state.get("u_t_ais", None)  # (n_pisos+1, nt)

    falt = []
    if dt is None: falt.append("dt")
    if u_fix is None: falt.append("u_t")
    if u_ais is None: falt.append("u_t_ais")
    if falt:
        st.error("❌ Faltan variables THA: " + ", ".join(falt))
        st.stop()

    dt = float(dt)
    u_fix = np.asarray(u_fix, float);  u_fix = u_fix if u_fix.ndim == 2 else u_fix[np.newaxis, :]
    u_ais = np.asarray(u_ais, float);  u_ais = u_ais if u_ais.ndim == 2 else u_ais[np.newaxis, :]

    y_ais_check = np.r_[0.0, alt_fix]
    if u_fix.shape[0] != len(alt_fix):
        st.error("❌ THA FIJA: u_t no calza con alturas.")
        st.stop()
    if u_ais.shape[0] != len(y_ais_check):
        st.error("❌ THA AISLADA: u_t_ais no calza con y_ais.")
        st.stop()

    if modo_tha_disp is None:
        modo_tha_disp = "Tiempo"

    t_fix = np.arange(u_fix.shape[1], dtype=float) * dt
    t_ais = np.arange(u_ais.shape[1], dtype=float) * dt

    if "Tiempo" in str(modo_tha_disp):
        tmax = float(min(t_fix[-1], t_ais[-1]))
        t_sel = st.slider(
            "Selecciona el tiempo t [s] (THA – heredado)",
            0.0, tmax,
            min(5.0, tmax),
            0.01,
            key="t_sel_derivas"
        )
        i_fix = int(np.argmin(np.abs(t_fix - t_sel)))
        i_ais = int(np.argmin(np.abs(t_ais - t_sel)))

        u_fix_levels = np.r_[0.0, u_fix[:, i_fix]]
        u_ais_levels = u_ais[:, i_ais]
        tag = f"THA (t ≈ {t_fix[i_fix]:.3f} s)"
    else:
        u_fix_levels = np.r_[0.0, np.max(np.abs(u_fix), axis=1)]
        u_ais_levels = np.max(np.abs(u_ais), axis=1)
        tag = "THA (máximos absolutos)"

# =============================================================================
# Calcular derivas reales
# =============================================================================
try:
    drift_fix_real, y_fix_story = _calc_drifts(u_fix_levels, y_fix)
    drift_ais_real, y_ais_story = _calc_drifts(u_ais_levels, y_ais)
except Exception as e:
    st.error(f"❌ Error calculando derivas: {e}")
    st.stop()

# =============================================================================
# Aplicar NEC si corresponde
# =============================================================================
if "NEC24" in tipo_deriva:
    drift_fix = (float(Cd) * drift_fix_real) / float(Ie)
    drift_ais = (float(Cd) * drift_ais_real) / float(Ie)
    xlabel_plot = "Deriva NEC24 Δ/h"
else:
    drift_fix = drift_fix_real
    drift_ais = drift_ais_real
    xlabel_plot = "Deriva Δ/h"

# Tablas
entrep_fix = ["0→1"] + [f"{i}→{i+1}" for i in range(1, len(drift_fix))]
entrep_ais = ["0→1"] + [f"{i}→{i+1}" for i in range(1, len(drift_ais))]

dfL = pd.DataFrame({
    "Entrepiso (nivel)": entrep_fix,
    "Altura sup [m]": np.round(y_fix_story, 3),
    "Deriva": np.round(drift_fix, 6),
    "Deriva [%]": np.round(drift_fix * 100.0, 3),
})

dfR = pd.DataFrame({
    "Entrepiso (nivel)": entrep_ais,
    "Altura sup [m]": np.round(y_ais_story, 3),
    "Deriva": np.round(drift_ais, 6),
    "Deriva [%]": np.round(drift_ais * 100.0, 3),
})

# =============================================================================
# Layout: tablas comprimidas + plots con líneas finas para muchos pisos
# =============================================================================
colL, colR = st.columns([1, 1], gap="large")

with colL:
    with st.container(border=True):
        st.subheader(f"🟦 FIJA – {tag}")
        with st.expander("📋 Ver tabla de derivas (FIJA)", expanded=False):
            _df_to_compact_table(dfL)
        drift_fix_plot = np.r_[0.0, drift_fix]
        y_fix_plot     = np.r_[0.0, y_fix_story]
        _plot_drift_sym(
            drift_fix_plot, y_fix_plot,
            "Derivas por entrepiso – FIJA",
            COLOR_FIX,
            xlabel=xlabel_plot,
            n_pisos_ref=n_pisos
        )

with colR:
    with st.container(border=True):
        st.subheader(f"🟩 AISLADA – {tag}")
        with st.expander("📋 Ver tabla de derivas (AISLADA)", expanded=False):
            _df_to_compact_table(dfR)
        drift_ais_plot = np.r_[0.0, drift_ais]
        y_ais_plot     = np.r_[0.0, y_ais_story]
        _plot_drift_sym(
            drift_ais_plot, y_ais_plot,
            "Derivas por entrepiso – AISLADA",
            COLOR_AIS,
            xlabel=xlabel_plot,
            n_pisos_ref=n_pisos
        )

st.success("✅ Derivas listas – heredan método y Sa desde desplazamientos; opción Deriva NEC24 incluida (límites NEC desactivados).")

# =============================================================================
# === BLOQUE 11: COMPARATIVO FINAL – FIJA vs AISLADA (SIN CONTROLES) ==========
# === 2x2: Cortantes | Desplazamientos
# ===      Derivas   | Resumen
# === Gráficos MONTADOS (FIJA vs AISLADA en el mismo eje)
# === Cada gráfico muestra SOLO su método (sin “mixto”)
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

st.markdown("## 📌 Comparativo final – FIJA vs AISLADA")

# ----------------------- Estilos -----------------------
BG            = "#2B3141"
COLOR_TEXT    = "#E8EDF2"
COLOR_GRID    = "#5B657A"

COLOR_FIX_V   = "#A8D5FF"   # FIJA cortantes
COLOR_AIS_V   = "#77DD77"   # AIS cortantes

COLOR_FIX_U   = "#FFE6A3"   # FIJA disp/deriva
COLOR_AIS_U   = "#77DD77"   # AIS disp/deriva

HALO = [pe.withStroke(linewidth=2.4, foreground=BG), pe.Normal()]

# ----------------------- Helpers -----------------------
def _dominant_to_right(x):
    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return x
    imax = int(np.argmax(np.abs(x)))
    s = np.sign(x[imax]) or 1.0
    return x * s

def _calc_drifts(u_levels, y_levels):
    u = np.asarray(u_levels, float).ravel()
    y = np.asarray(y_levels, float).ravel()
    if len(u) != len(y):
        raise ValueError("u_levels y y_levels no coinciden.")
    dy = np.diff(y); du = np.diff(u)
    if np.any(np.abs(dy) < 1e-12):
        raise ValueError("Hay alturas repetidas (dy=0).")
    return du/dy, y[1:]

def _stairs_xy_etabs_one_side(V_story, y_levels):
    V_story  = np.asarray(V_story, float).ravel()
    y_levels = np.asarray(y_levels, float).ravel()
    if len(y_levels) != len(V_story) + 1:
        raise ValueError("Cortantes: y_levels debe ser n+1 y V_story n.")
    if len(V_story) == 0:
        return np.array([]), np.array([])

    Vp = _dominant_to_right(V_story)
    xs, ys = [], []
    xs.append(Vp[0]); ys.append(y_levels[0])
    for i in range(len(Vp)):
        xs += [Vp[i], Vp[i]]
        ys += [y_levels[i], y_levels[i+1]]
        if i < len(Vp)-1:
            xs += [Vp[i+1]]
            ys += [y_levels[i+1]]
    return np.asarray(xs, float), np.asarray(ys, float)

def _pct_change(base, new):
    base = float(base); new = float(new)
    if abs(base) < 1e-12:
        return np.nan
    return 100.0*(new-base)/base

def _df_height(df: pd.DataFrame, row_h=35, header_h=38, pad=14, min_h=140, max_h=380):
    n = int(max(len(df), 1))
    h = header_h + pad + row_h * n
    return int(max(min_h, min(max_h, h)))

def _style_by_npiso(n_pisos: int):
    n = int(max(1, n_pisos))
    lw_line  = float(np.clip(2.2 - 0.03*(n-1), 0.85, 2.2))
    lw_axis  = float(np.clip(1.1 - 0.01*(n-1), 0.65, 1.1))
    ms       = float(np.clip(5.2 - 0.08*(n-1), 2.2, 5.2))
    fs_tick  = float(np.clip(10.0 - 0.08*(n-1), 7.0, 10.0))
    fs_lab   = float(np.clip(10.5 - 0.07*(n-1), 7.6, 10.5))
    fs_title = float(np.clip(12.0 - 0.06*(n-1), 9.0, 12.0))
    fs_leg   = float(np.clip(10.0 - 0.08*(n-1), 7.2, 10.0))
    return lw_line, lw_axis, ms, fs_tick, fs_lab, fs_title, fs_leg

def _common_axes_style(ax, title, xlabel, ylabel, lw_axis, fs_tick, fs_lab, fs_title):
    ax.axvline(0.0, color=COLOR_GRID, lw=lw_axis, alpha=0.62)
    ax.set_xlabel(xlabel, color=COLOR_TEXT, fontsize=fs_lab)
    ax.set_ylabel(ylabel, color=COLOR_TEXT, fontsize=fs_lab)
    ax.set_title(title, color=COLOR_TEXT, fontweight="bold", fontsize=fs_title)
    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
    ax.tick_params(colors=COLOR_TEXT, labelsize=fs_tick)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def _finish_legend(ax, fs_leg):
    leg = ax.legend(frameon=True, fontsize=fs_leg)
    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_edgecolor(COLOR_GRID)

def _set_xlim_positive(ax, arrA, arrB):
    arrA = np.asarray(arrA, float).ravel()
    arrB = np.asarray(arrB, float).ravel()
    vmax = float(np.max(np.abs(np.r_[arrA, arrB]))) if (arrA.size or arrB.size) else 1.0
    vmax = 1.0 if vmax <= 0 else vmax
    ax.set_xlim(0.0, 1.10*vmax)

def _plot_story_shear_overlay(V_fix_story, V_ais_story, y_levels, title, style):
    lw_line, lw_axis, ms, fs_tick, fs_lab, fs_title, fs_leg = style
    xsF, ysF = _stairs_xy_etabs_one_side(V_fix_story, y_levels)
    xsA, ysA = _stairs_xy_etabs_one_side(V_ais_story, y_levels)

    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.plot(xsF, ysF, "-", color=COLOR_FIX_V, lw=lw_line, label="FIJA")
    ax.plot(xsA, ysA, "-", color=COLOR_AIS_V, lw=lw_line, label="AISLADA")

    VpF = _dominant_to_right(V_fix_story)
    VpA = _dominant_to_right(V_ais_story)
    ax.plot(np.r_[VpF[0], VpF][:len(y_levels)], y_levels, "o", color=COLOR_FIX_V, ms=ms)
    ax.plot(np.r_[VpA[0], VpA][:len(y_levels)], y_levels, "o", color=COLOR_AIS_V, ms=ms)

    _common_axes_style(ax, title, "Cortante [Tf]", "Altura [m]", lw_axis, fs_tick, fs_lab, fs_title)
    _set_xlim_positive(ax, VpF, VpA)
    _finish_legend(ax, fs_leg)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

def _plot_profile_overlay(U_fix_levels, U_ais_levels, y_levels, title, style):
    lw_line, lw_axis, ms, fs_tick, fs_lab, fs_title, fs_leg = style
    U_fix = _dominant_to_right(U_fix_levels)
    U_ais = _dominant_to_right(U_ais_levels)

    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.plot(U_fix, y_levels, "-o", color=COLOR_FIX_U, lw=lw_line, ms=ms, label="FIJA")
    ax.plot(U_ais, y_levels, "-o", color=COLOR_AIS_U, lw=lw_line, ms=ms, label="AISLADA")

    _common_axes_style(ax, title, "Desplazamiento [m]", "Altura [m]", lw_axis, fs_tick, fs_lab, fs_title)
    _set_xlim_positive(ax, U_fix, U_ais)
    _finish_legend(ax, fs_leg)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

def _plot_drift_overlay(drift_fix, drift_ais, y_story, title, style):
    lw_line, lw_axis, ms, fs_tick, fs_lab, fs_title, fs_leg = style
    dF = _dominant_to_right(drift_fix)
    dA = _dominant_to_right(drift_ais)

    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.plot(dF, y_story, "-o", color=COLOR_FIX_U, lw=lw_line, ms=ms, label="FIJA")
    ax.plot(dA, y_story, "-o", color=COLOR_AIS_U, lw=lw_line, ms=ms, label="AISLADA")

    _common_axes_style(ax, title, "Deriva Δ/h", "Altura [m]", lw_axis, fs_tick, fs_lab, fs_title)
    _set_xlim_positive(ax, dF, dA)
    _finish_legend(ax, fs_leg)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ------------------ Leer heredados ------------------
alt_fix = np.asarray(st.session_state.get("alturas", []), float).ravel()
if alt_fix.size == 0:
    st.error("❌ Falta st.session_state['alturas'].")
    st.stop()

y_levels = np.r_[0.0, alt_fix]
n_pisos = len(alt_fix)

V_fix_story  = np.asarray(st.session_state.get("cmp_V_fix_story", []), float).ravel()
V_ais_story  = np.asarray(st.session_state.get("cmp_V_ais_story", []), float).ravel()
U_fix_levels = np.asarray(st.session_state.get("cmp_U_fix_levels", []), float).ravel()
U_ais_levels = np.asarray(st.session_state.get("cmp_U_ais_levels", []), float).ravel()

if not (len(V_fix_story)==n_pisos and len(V_ais_story)==n_pisos and
        len(U_fix_levels)==len(y_levels) and len(U_ais_levels)==len(y_levels)):
    st.error("❌ Faltan o no calzan los heredados cmp_*.")
    st.stop()

# ------------------ Etiquetas (SIN mixtos) ------------------
tag_shear = str(st.session_state.get("cmp_tag_shear", ""))
tag_disp  = str(st.session_state.get("cmp_tag_disp", ""))

# Estilo adaptativo
STYLE = _style_by_npiso(n_pisos)

# Derivas desde desplazamientos
drift_fix, y_story = _calc_drifts(U_fix_levels, y_levels)
drift_ais, _       = _calc_drifts(U_ais_levels, y_levels)
drift_fix_plot = np.r_[0.0, drift_fix]
drift_ais_plot = np.r_[0.0, drift_ais]
y_story_plot   = np.r_[0.0, y_story]

# ------------------ Layout 2x2 ------------------
topL, topR = st.columns(2, gap="large")
botL, botR = st.columns(2, gap="large")

with topL:
    with st.container(border=True):
        st.subheader("🧱 Cortantes")
        _plot_story_shear_overlay(
            V_fix_story, V_ais_story, y_levels,
            f"Cortantes por piso – {tag_shear}",
            STYLE
        )

with topR:
    with st.container(border=True):
        st.subheader("🟦 Desplazamientos")
        _plot_profile_overlay(
            U_fix_levels, U_ais_levels, y_levels,
            f"Desplazamientos por nivel – {tag_disp}",
            STYLE
        )

with botL:
    with st.container(border=True):
        st.subheader("📐 Derivas")
        _plot_drift_overlay(
            drift_fix_plot, drift_ais_plot, y_story_plot,
            f"Derivas por entrepiso – {tag_disp}",
            STYLE
        )

with botR:
    with st.container(border=True):
        st.subheader("📊 Resumen técnico – comparación")

        Vb_fix = float(V_fix_story[0])
        Vb_ais = float(V_ais_story[0])
        pVb = _pct_change(Vb_fix, Vb_ais)

        dmax_fix = float(np.max(np.abs(drift_fix_plot)))
        dmax_ais = float(np.max(np.abs(drift_ais_plot)))
        pDmax = _pct_change(dmax_fix, dmax_ais)

        u_iso = float(U_ais_levels[0])
        u_sup_fix = float(U_fix_levels[-1] - U_fix_levels[0])
        u_sup_ais = float(U_ais_levels[-1] - U_ais_levels[0])
        pUsup = _pct_change(u_sup_fix, u_sup_ais)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Cortante basal [Tf] (AIS vs FIJA)", f"{Vb_ais:.3f}",
                      f"{pVb:+.1f}%" if np.isfinite(pVb) else "—")
        with c2:
            st.metric("Deriva máxima (AIS vs FIJA)", f"{dmax_ais:.6g}",
                      f"{pDmax:+.1f}%" if np.isfinite(pDmax) else "—")
        with c3:
            st.metric("Desplazamiento en la base [m]", f"{u_iso:.4f}", "aislador")

        df = pd.DataFrame({
            "Métrica": [
                "Cortante basal [Tf]",
                "Deriva máxima",
                "Despl. superestructura (techo - base) [m]",
                "Desplazamiento en la base (aislador) [m]"
            ],
            "FIJA":    [Vb_fix, dmax_fix, u_sup_fix, 0.0],
            "AISLADA": [Vb_ais, dmax_ais, u_sup_ais, u_iso],
            "% cambio (AIS vs FIJA)": [
                f"{pVb:+.1f}%" if np.isfinite(pVb) else "—",
                f"{pDmax:+.1f}%" if np.isfinite(pDmax) else "—",
                f"{pUsup:+.1f}%" if np.isfinite(pUsup) else "—",
                "—"
            ],
            "Lectura": [
                "↓ mejor (menor fuerza)" if np.isfinite(pVb) and pVb < 0 else "↑ mayor fuerza",
                "↓ mejor (menor demanda)" if np.isfinite(pDmax) and pDmax < 0 else "↑ mayor demanda",
                "↓ mejor (menor demanda en marco)" if np.isfinite(pUsup) and pUsup < 0 else "↑ mayor en marco",
                "Costo del aislamiento (esperable)"
            ]
        })

        st.dataframe(df, use_container_width=True, hide_index=True, height=_df_height(df))

        st.caption(
            "En aislación es normal que el desplazamiento total se concentre en la base (aislador). "
            "Lo crítico es reducir cortantes y derivas en la superestructura."
        )














