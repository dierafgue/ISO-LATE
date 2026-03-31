# =============================================================================
# APP STREAMLIT – ISO-LATE - DESARROLLADO POR DIEGO GUERRERO MDI.
# =============================================================================
import streamlit as st
import numpy as np
from funciones_usuario import *

# =============================================================================
# === CONFIGURACIÓN INICIAL ==================================================
# =============================================================================
st.set_page_config(
    page_title="ISO-LATE",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# =============================================================================
# === CSS (solo una vez) =====================================================
# =============================================================================
@st.cache_data(show_spinner=False)
def _get_css() -> str:
    return """
    <style>
    /* =========================
       OCULTAR HEADER STREAMLIT
    ========================= */
    header[data-testid="stHeader"]{
        height: 0px !important;
        visibility: hidden !important;
    }

    div[data-testid="stToolbar"]{
        height: 0px !important;
        visibility: hidden !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* =========================
       ESTILO GLOBAL
    ========================= */
    body {
        background-color: #f6f8fb;
        color: #222831;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 0.20rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2.20rem !important;
        padding-right: 2.20rem !important;
        max-width: 100% !important;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    section[data-testid="stMain"] {
        margin-left: 0 !important;
        width: 100% !important;
    }

    .stButton > button {
        border-radius: 8px;
        background-color: #30475e;
        color: white;
        border: none;
        font-weight: 500;
        padding: 0.5rem 1.2rem;
    }

    .stButton > button:hover {
        background-color: #3e5a78;
    }

    input, textarea, select {
        border-radius: 6px !important;
    }

    h1 {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.10 !important;
    }

    div[role="radiogroup"] {
        gap: 0.22rem !important;
        margin-top: -0.35rem !important;
        margin-bottom: -0.65rem !important;
        justify-content: flex-start !important;
    }

    div[role="radiogroup"] label {
        font-size: 0.86rem !important;
        padding: 0rem 0.04rem !important;
    }

    hr {
        margin: 0.10rem 0 0.20rem 0 !important;
    }

    /* =========================
       HERO / HEADER SUPERIOR
    ========================= */
    .hero-card {
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
        box-shadow: none;
        margin: 0 0 0.01rem 0;
    }

    .hero-stack {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: flex-start;
        gap: 0;
        text-align: left;
        margin: 0;
        padding: 0;
    }

    .hero-title {
        font-size: 3.10rem;
        font-weight: 800;
        color: #2d3142;
        letter-spacing: -0.03em;
        line-height: 1.0;
        margin: 0;
        padding: 0;
    }

    .hero-logo-wrap {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        margin: -0.20rem 0 -1.00rem 0;
        padding: 0;
    }

    .hero-lang {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        width: 100%;
        margin: -0.35rem 0 -0.40rem 0;
        padding: 0;
    }

    .hero-divider {
        height: 1px;
        background: #e3e9f1;
        margin: -0.35rem 0 0.12rem 0;
    }

    /* =========================
       COLUMNAS EXTERNAS
    ========================= */
    [data-testid="column"] {
        display: flex;
        align-items: stretch;
    }

    [data-testid="column"] > div {
        width: 100%;
        height: 100%;
    }

    /* =========================
       TARJETAS
    ========================= */
    .iso-card {
        background: #ffffff;
        border: 1px solid #e6ebf2;
        border-radius: 14px;
        padding: 1.00rem 1.10rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
        margin-top: 0.05rem;
        margin-bottom: 0.35rem;
        min-height: 230px;
        display: flex;
        flex-direction: column;
    }

    .iso-card h3 {
        margin-top: 0rem !important;
        margin-bottom: 0.55rem !important;
        color: #243447;
        font-size: 1.08rem !important;
    }

    .iso-muted {
        color: #5f6b7a;
        font-size: 0.95rem;
        line-height: 1.68;
        text-align: justify;
    }

    .iso-section-sep {
        height: 1px;
        background: #e9eef5;
        margin: 0.75rem 0 0.70rem 0;
    }
    
    .iso-subtitle {
        color: #243447;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.30rem;
    }
    
    .iso-license-box {
        margin-top: 0.10rem;
    }
    
    .iso-license-text {
        color: #5f6b7a;
        font-size: 0.88rem;
        line-height: 1.55;
        text-align: justify;
    }
    
    .iso-license-text + .iso-license-text {
        margin-top: 0.38rem;
    }

    /* =========================
       TARJETA DERECHA
    ========================= */
    .iso-author-grid {
        display: grid;
        grid-template-columns: 1.55fr 0.78fr;
        gap: 0.65rem;
        align-items: start;
        height: 100%;
    }

    .iso-author-left {
        min-width: 0;
        padding-right: 0.20rem;
    }

    .iso-author-right {
        min-width: 0;
        border-left: 1px solid #e6ebf2;
        padding-left: 0.70rem;
    }

    .iso-info-box {
        color: #243447;
        font-size: 0.95rem;
        line-height: 1.48;
    }

    .iso-info-row {
        margin-bottom: 0.38rem;
    }

    .iso-info-label {
        font-weight: 700;
        color: #243447;
    }

    .iso-contact-title {
        font-size: 0.94rem;
        font-weight: 700;
        color: #243447;
        margin-bottom: 0.45rem;
    }

    .iso-contact-wrap {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
        align-items: flex-start;
    }

    /* =========================
       BADGES
    ========================= */
    .iso-badge, .iso-email {
        display: inline-flex;
        align-items: center;
        gap: 0.36rem;
        padding: 0.28rem 0.58rem;
        border-radius: 7px;
        font-weight: 600;
        font-size: 0.84rem;
        line-height: 1.15;
        transition: 0.18s ease;
        white-space: nowrap;
        text-decoration: none !important;
    }

    .iso-badge {
        border: 1px solid #d0d7de;
        background: #f6f8fa;
        color: #24292f !important;
    }

    .iso-badge:hover {
        background: #eef2f6;
        border-color: #bfc8d1;
        color: #24292f !important;
    }

    .iso-email {
        border: 1px solid #d0d7de;
        background: #ffffff;
        color: #30475e !important;
    }

    .iso-email:hover {
        background: #f8fafc;
        border-color: #bfc8d1;
        color: #243447 !important;
    }

    .iso-github-icon,
    .iso-mail-icon {
        width: 13px;
        height: 13px;
        display: inline-block;
        vertical-align: middle;
        fill: currentColor;
        flex-shrink: 0;
    }

    @media (max-width: 980px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .hero-title {
            font-size: 2.35rem;
        }

        .hero-logo-wrap {
            margin: -0.10rem 0 -0.75rem 0;
        }

        .hero-lang {
            margin: -0.90rem 0 -0.40rem 0;
        }

        .iso-author-grid {
            grid-template-columns: 1fr;
            gap: 0.65rem;
        }

        .iso-author-right {
            border-left: none;
            padding-left: 0rem;
            border-top: 1px solid #e6ebf2;
            padding-top: 0.60rem;
            margin-top: 0.05rem;
        }

        .iso-card {
            height: auto;
            min-height: auto;
        }
    }
    </style>
    """

st.markdown(_get_css(), unsafe_allow_html=True)

# =============================================================================
# === IDIOMA (EN por defecto) ================================================
# =============================================================================
if "lang" not in st.session_state:
    st.session_state.lang = "en"

T = {
    "en": {
        "language": "Language",
        "english": "English",
        "spanish": "Spanish",

        "intro_title": "About ISO-LATE",
        "intro_text": (
            "ISO-LATE is an interactive structural engineering application for "
            "simulating, analyzing, and comparing the seismic response of 2D frame "
            "structures with fixed base and base isolation (LRB). It includes modal "
            "analysis, response spectrum analysis (RSA), time history analysis (THA), "
            "and NEC-24 spectrum tools."
        ),

        "license_title": "License",
        "license_text": (
            "ISO-LATE is distributed under the MIT License. "
            "Copyright (c) 2026 Diego Rafael Guerrero Carrillo."
        ),
        "license_note": (
            "Permission is granted to use, copy, modify, and distribute this software, "
            "subject to the terms of the MIT License."
        ),

        "info_title": "Author information",
        "author": "Author",
        "degree": "Degree",
        "university": "University",
        "program": "Program",
        "degree_value": "Civil Engineer – MDI",
        "program_value": "Master’s Degree in Earthquake-Resistant Structural Design",
        "contact_title": "Find me on:",
        "github_badge": "Follow",
        "mail_badge": "Mail",
    },

    "es": {
        "language": "Idioma",
        "english": "Inglés",
        "spanish": "Español",

        "intro_title": "Acerca de ISO-LATE",
        "intro_text": (
            "ISO-LATE es una aplicación interactiva de ingeniería estructural para "
            "simular, analizar y comparar la respuesta sísmica de pórticos 2D con base "
            "fija y aislamiento sísmico en la base (LRB). Incluye análisis modal, "
            "análisis espectral (RSA), análisis tiempo historia (THA) y herramientas "
            "del espectro NEC-24."
        ),

        "license_title": "Licencia",
        "license_text": (
            "ISO-LATE se distribuye bajo la Licencia MIT. "
            "Copyright (c) 2026 Diego Rafael Guerrero Carrillo."
        ),
        "license_note": (
            "Se concede permiso para usar, copiar, modificar y distribuir este software, "
            "conforme a los términos de la Licencia MIT."
        ),
        
        "info_title": "Información del autor",
        "author": "Autor",
        "degree": "Formación",
        "university": "Universidad",
        "program": "Programa",
        "degree_value": "Ingeniero Civil – MDI",
        "program_value": "Maestría en Diseño de Estructuras Sismorresistentes",
        "contact_title": "Encuéntrame en:",
        "github_badge": "Seguir",
        "mail_badge": "Mail",
    },
}

def tr(key: str) -> str:
    return T.get(st.session_state.lang, T["en"]).get(key, key)

# =============================================================================
# ✅ HEADER SUPERIOR ==========================================================
# =============================================================================
st.markdown('<div class="hero-card">', unsafe_allow_html=True)
st.markdown('<div class="hero-stack">', unsafe_allow_html=True)

# Título
st.markdown(
    '<div class="hero-title">ISO-LATE</div>',
    unsafe_allow_html=True
)

# Selector de idioma
st.markdown('<div class="hero-lang">', unsafe_allow_html=True)
lang = st.radio(
    label="",
    options=["EN", "ES"],
    index=0 if st.session_state.lang == "en" else 1,
    horizontal=True,
    label_visibility="collapsed",
    key="main_lang_selector"
)
st.markdown('</div>', unsafe_allow_html=True)

# Cambio de idioma
new_lang = "en" if lang == "EN" else "es"
if new_lang != st.session_state.lang:
    st.session_state.lang = new_lang
    st.rerun()

# Línea separadora
st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# === BLOQUE INICIAL =========================================================
# =============================================================================
col1, col2 = st.columns([1.60, 1.40], gap="large")

github_icon_svg = """
<svg class="iso-github-icon" viewBox="0 0 16 16" aria-hidden="true">
<path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38
0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95
0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82
.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82
.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15
0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.001 8.001 0 0 0 16 8
c0-4.42-3.58-8-8-8Z"></path>
</svg>
"""

mail_icon_svg = """
<svg class="iso-mail-icon" viewBox="0 0 16 16" aria-hidden="true">
<path d="M1.75 3A1.75 1.75 0 0 0 0 4.75v6.5C0 12.216.784 13 1.75 13h12.5A1.75 1.75 0 0 0 16 11.25v-6.5A1.75 1.75 0 0 0 14.25 3H1.75ZM1.5 4.75c0-.138.112-.25.25-.25h12.5c.138 0 .25.112.25.25v.243L8 8.94 1.5 4.993V4.75Zm0 1.997 4.906 2.978L1.5 11.91V6.747Zm1.279 4.753 4.935-2.196a.75.75 0 0 1 .572 0l4.935 2.196H2.779Zm11.721-.59-4.906-2.185L14.5 6.747v4.163Z"></path>
</svg>
"""

about_html = (
    f'<div class="iso-card">'
        f'<h3>{tr("intro_title")}</h3>'
        f'<div class="iso-muted">{tr("intro_text")}</div>'

        f'<div class="iso-section-sep"></div>'

        f'<div class="iso-license-box">'
            f'<div class="iso-subtitle">{tr("license_title")}</div>'
            f'<div class="iso-license-text">'
                f'{tr("license_text")}<br>{tr("license_note")}'
            f'</div>'
        f'</div>'
    f'</div>'
)

author_html = (
    f'<div class="iso-card">'
    f'<h3>{tr("info_title")}</h3>'
    f'<div class="iso-author-grid">'
        f'<div class="iso-author-left">'
            f'<div class="iso-info-box">'
                f'<div class="iso-info-row"><span class="iso-info-label">{tr("author")}:</span> Diego Rafael Guerrero Carrillo</div>'
                f'<div class="iso-info-row"><span class="iso-info-label">{tr("degree")}:</span> {tr("degree_value")}</div>'
                f'<div class="iso-info-row"><span class="iso-info-label">{tr("university")}:</span> Pontificia Universidad Católica del Ecuador</div>'
                f'<div class="iso-info-row"><span class="iso-info-label">{tr("program")}:</span> {tr("program_value")}</div>'
            f'</div>'
        f'</div>'
        f'<div class="iso-author-right">'
            f'<div class="iso-contact-title">{tr("contact_title")}</div>'
            f'<div class="iso-contact-wrap">'
                f'<a href="https://github.com/dierafgue" target="_blank" class="iso-badge">{github_icon_svg}<span>{tr("github_badge")}</span></a>'
                f'<a href="mailto:DRAFAELGUE@HOTMAIL.COM" class="iso-email">{mail_icon_svg}<span>{tr("mail_badge")}</span></a>'
            f'</div>'
        f'</div>'
    f'</div>'
    f'</div>'
)

with col1:
    st.markdown(about_html, unsafe_allow_html=True)

with col2:
    st.markdown(author_html, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# === SECCIÓN 1: PARÁMETROS GENERALES DEL MODELO (DISEÑO SIMÉTRICO) ==========
# =============================================================================

from funciones_usuario import (
    seccion_rectangular_cm_to_SI,
    seccion_AI_cm_to_SI,
    build_param_estruct
)

# -------------------------------------------------------------------------
# ✅ Textos EN/ES (solo para este bloque)
# -------------------------------------------------------------------------
T["en"].update({
    "hdr_general": "General structural model parameters",
    "txt_general": "Define geometry, sections and material properties for the base model.",

    "sub_geo": "Geometry",
    "n_floors": "Number of stories",
    "n_bays": "Number of bays",
    "ranges": "Allowed ranges: stories 1–20 | bays 1–6",
    "bay_length": "Bay length [m]",
    "h_first": "1st story height [m]",
    "h_rest": "Typical story height [m]",

    "sub_sec": "Section properties",
    "adv_mode": "Advanced mode",
    "basic_mode": "Basic mode (dimensions in cm)",
    "advanced_mode": "Advanced mode (properties in cm² / cm⁴)",
    "col": "Column",
    "beam": "Beam",
    "b_col": "Column width [cm]",
    "h_col": "Column depth [cm]",
    "b_beam": "Beam width [cm]",
    "h_beam": "Beam depth [cm]",
    "A_col": "Column area [cm²]",
    "I_col": "Column inertia [cm⁴]",
    "A_beam": "Beam area [cm²]",
    "I_beam": "Beam inertia [cm⁴]",

    "sub_mat": "Material and loads",
    "E": "Elastic modulus E [Tf/m²]",
    "gamma": "Unit weight [Tf/m³]",
    "loads": "Loads",
    "dl": "Additional dead load [Tf/m]",
    "damp": "Damping ζ (%)",

    "help_suffix": "Tip: use the helper (ⓘ).",
    "help_n_floors": "Total number of stories for the frame model.",
    "help_n_bays": "Total number of bays (spans) for the frame model.",
    "help_bay_length": "Span length (center-to-center) in meters.",
    "help_h_first": "Height of the first story in meters.",
    "help_h_rest": "Height of remaining stories in meters.",
    "help_adv_mode": "Switch between basic (b×h) and advanced (A, I) section input.",
    "help_b_col": "Column width b in centimeters.",
    "help_h_col": "Column depth h in centimeters.",
    "help_b_beam": "Beam width b in centimeters.",
    "help_h_beam": "Beam depth h in centimeters.",
    "help_A_col": "Column area in cm² (advanced mode).",
    "help_I_col": "Column second moment of area (inertia) in cm⁴ (advanced mode).",
    "help_A_beam": "Beam area in cm² (advanced mode).",
    "help_I_beam": "Beam inertia in cm⁴ (advanced mode).",
    "help_E": "Elastic modulus of the material (consistent units).",
    "help_gamma": "Unit weight (specific weight) with consistent units.",
    "help_dl": "Uniform additional dead load per meter.",
    "help_damp": "Equivalent viscous damping ratio in percent.",
})

T["es"].update({
    "hdr_general": "Parámetros generales del modelo estructural",
    "txt_general": "Define geometría, secciones y propiedades del material del modelo base.",

    "sub_geo": "Geometría",
    "n_floors": "Número de pisos",
    "n_bays": "Número de vanos",
    "ranges": "Rangos permitidos: pisos 1–20 | vanos 1–6",
    "bay_length": "Longitud de vano [m]",
    "h_first": "Altura 1er piso [m]",
    "h_rest": "Altura pisos restantes [m]",

    "sub_sec": "Propiedades de secciones",
    "adv_mode": "Modo avanzado",
    "basic_mode": "Modo básico (dimensiones en cm)",
    "advanced_mode": "Modo avanzado (propiedades en cm² / cm⁴)",
    "col": "Columna",
    "beam": "Viga",
    "b_col": "Base columna [cm]",
    "h_col": "Altura columna [cm]",
    "b_beam": "Base viga [cm]",
    "h_beam": "Altura viga [cm]",
    "A_col": "Área columna [cm²]",
    "I_col": "Inercia columna [cm⁴]",
    "A_beam": "Área viga [cm²]",
    "I_beam": "Inercia viga [cm⁴]",

    "sub_mat": "Material y cargas",
    "E": "Módulo E [Tf/m²]",
    "gamma": "Peso específico [Tf/m³]",
    "loads": "Cargas",
    "dl": "Sobrecarga muerta [Tf/m]",
    "damp": "Amortiguamiento ζ (%)",

    "help_suffix": "Tip: usa el helper (ⓘ).",
    "help_n_floors": "Número total de pisos del pórtico.",
    "help_n_bays": "Número total de vanos (luces) del pórtico.",
    "help_bay_length": "Longitud de luz (centro a centro) en metros.",
    "help_h_first": "Altura del primer piso en metros.",
    "help_h_rest": "Altura de los pisos restantes en metros.",
    "help_adv_mode": "Cambia entre ingreso básico (b×h) y avanzado (A, I).",
    "help_b_col": "Base b de la columna en centímetros.",
    "help_h_col": "Altura h de la columna en centímetros.",
    "help_b_beam": "Base b de la viga en centímetros.",
    "help_h_beam": "Altura h de la viga en centímetros.",
    "help_A_col": "Área de la columna en cm² (modo avanzado).",
    "help_I_col": "Momento de inercia de la columna en cm⁴ (modo avanzado).",
    "help_A_beam": "Área de la viga en cm² (modo avanzado).",
    "help_I_beam": "Inercia de la viga en cm⁴ (modo avanzado).",
    "help_E": "Módulo de elasticidad del material (unidades consistentes).",
    "help_gamma": "Peso específico con unidades consistentes.",
    "help_dl": "Sobrecarga muerta uniforme por metro.",
    "help_damp": "Amortiguamiento viscoso equivalente en porcentaje.",
})

def H(key: str) -> str:
    return tr(key)

# -------------------------------------------------------------------------
# ✅ Getter seguro (evita float(None) al cambiar modos)
# -------------------------------------------------------------------------
def ss_num(key: str, default: float) -> float:
    try:
        p = st.session_state.get("param_estruct", {})
        v = p.get(key, default) if isinstance(p, dict) else default
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)

# -------------------------------------------------------------------------
# 🧩 UI
# -------------------------------------------------------------------------
st.header(f"📋 {tr('hdr_general')}")
st.markdown(tr("txt_general"))

col_geo, col_sec, col_mat = st.columns(3, gap="large")

# -------------------------------------------------------------------------
# ⚙️ GEOMETRÍA
# -------------------------------------------------------------------------
with col_geo:
    st.subheader(f"⚙️ {tr('sub_geo')}")
    c1, c2 = st.columns(2)

    n_pisos = c1.number_input(
        tr("n_floors"),
        min_value=1, max_value=20,
        value=int(ss_num("n_pisos", 2)),
        step=1,
        help=tr("help_n_floors"),
    )

    n_vanos = c2.number_input(
        tr("n_bays"),
        min_value=1, max_value=6,
        value=int(ss_num("n_vanos", 1)),
        step=1,
        help=tr("help_n_bays"),
    )

    st.caption(tr("ranges"))

    l_vano = st.number_input(
        tr("bay_length"),
        min_value=1.0,
        value=ss_num("l_vano", 5.0),
        step=0.5,
        help=tr("help_bay_length"),
    )

    h_piso_1 = round(
        st.number_input(
            tr("h_first"),
            min_value=2.0,
            value=ss_num("h_piso_1", 4.0),
            step=0.1,
            help=tr("help_h_first"),
        ),
        6
    )

    h_piso_restantes = round(
        st.number_input(
            tr("h_rest"),
            min_value=2.0,
            value=ss_num("h_piso_restantes", 3.0),
            step=0.1,
            help=tr("help_h_rest"),
        ),
        6
    )

# -------------------------------------------------------------------------
# 📏 SECCIONES (Básico / Avanzado)
# -------------------------------------------------------------------------
with col_sec:
    st.subheader(f"📏 {tr('sub_sec')}")
    modo_avanzado = st.checkbox(
        f"🔧 {tr('adv_mode')}",
        value=bool(st.session_state.get("param_estruct", {}).get("modo_avanzado", False)),
        help=tr("help_adv_mode"),
    )

    if not modo_avanzado:
        st.caption(tr("basic_mode"))

        st.markdown(f"#### 🧱 {tr('col')}")
        cc1, cc2 = st.columns(2)
        b_col = cc1.number_input(
            tr("b_col"),
            value=ss_num("b_col_cm", 50.0),
            step=0.5,
            help=tr("help_b_col"),
        )
        h_col = cc2.number_input(
            tr("h_col"),
            value=ss_num("h_col_cm", 50.0),
            step=0.5,
            help=tr("help_h_col"),
        )

        st.markdown(f"#### 🪵 {tr('beam')}")
        cv1, cv2 = st.columns(2)
        b_viga = cv1.number_input(
            tr("b_beam"),
            value=ss_num("b_viga_cm", 30.0),
            step=0.5,
            help=tr("help_b_beam"),
        )
        h_viga = cv2.number_input(
            tr("h_beam"),
            value=ss_num("h_viga_cm", 50.0),
            step=0.5,
            help=tr("help_h_beam"),
        )

        # SI (m², m⁴) para el modelo
        A_col, I_col   = seccion_rectangular_cm_to_SI(b_col, h_col)
        A_viga, I_viga = seccion_rectangular_cm_to_SI(b_viga, h_viga)

        # ✅ Guardar también equivalentes en cm² / cm⁴ (evita float(None) al pasar a avanzado)
        A_col_cm2  = float(b_col * h_col)
        I_col_cm4  = float(b_col * (h_col**3) / 12.0)
        A_viga_cm2 = float(b_viga * h_viga)
        I_viga_cm4 = float(b_viga * (h_viga**3) / 12.0)

        b_col_cm, h_col_cm = float(b_col), float(h_col)
        b_viga_cm, h_viga_cm = float(b_viga), float(h_viga)

    else:
        st.caption(tr("advanced_mode"))

        st.markdown(f"#### 🧱 {tr('col')}")
        ca1, ca2 = st.columns(2)
        A_col_cm2 = ca1.number_input(
            tr("A_col"),
            value=ss_num("A_col_cm2", 2500.00),
            step=10.0,
            help=tr("help_A_col"),
        )
        I_col_cm4 = ca2.number_input(
            tr("I_col"),
            value=ss_num("I_col_cm4", 520833.33),
            step=100.0,
            help=tr("help_I_col"),
        )

        st.markdown(f"#### 🪵 {tr('beam')}")
        cb1, cb2 = st.columns(2)
        A_viga_cm2 = cb1.number_input(
            tr("A_beam"),
            value=ss_num("A_viga_cm2", 1500.00),
            step=10.0,
            help=tr("help_A_beam"),
        )
        I_viga_cm4 = cb2.number_input(
            tr("I_beam"),
            value=ss_num("I_viga_cm4", 312500.00),
            step=100.0,
            help=tr("help_I_beam"),
        )

        # SI (m², m⁴) para el modelo
        A_col, I_col   = seccion_AI_cm_to_SI(A_col_cm2, I_col_cm4)
        A_viga, I_viga = seccion_AI_cm_to_SI(A_viga_cm2, I_viga_cm4)

        # En avanzado no se usan dimensiones b/h
        b_col_cm = h_col_cm = b_viga_cm = h_viga_cm = None

# -------------------------------------------------------------------------
# 🧱 MATERIAL Y CARGAS
# -------------------------------------------------------------------------
with col_mat:
    st.subheader(f"🧱 {tr('sub_mat')}")

    cm1, cm2 = st.columns(2)
    E = cm1.number_input(
        tr("E"),
        value=ss_num("E", 2534563.54),
        step=10000.0,
        help=tr("help_E"),
    )

    peso_especifico = cm2.number_input(
        tr("gamma"),
        value=ss_num("peso_especifico", 2.4028),
        step=0.0001,
        format="%.4f",
        help=tr("help_gamma"),
    )

    st.markdown(f"#### ⚖️ {tr('loads')}")
    sobrecarga_muerta = st.number_input(
        tr("dl"),
        value=ss_num("sobrecarga_muerta", 0.0),
        step=1.0,
        help=tr("help_dl"),
    )

    amortiguamiento = st.number_input(
        tr("damp"),
        min_value=0.0, max_value=10.0,
        value=ss_num("amortiguamiento", 0.05) * 100.0,
        step=0.5,
        help=tr("help_damp"),
    )

# -------------------------------------------------------------------------
# ✅ Almacenamiento
# -------------------------------------------------------------------------
params_nuevos = build_param_estruct(
    n_pisos=n_pisos,
    n_vanos=n_vanos,
    l_vano=l_vano,
    h_piso_1=h_piso_1,
    h_piso_restantes=h_piso_restantes,
    E=E,
    A_col=A_col,
    I_col=I_col,
    A_viga=A_viga,
    I_viga=I_viga,
    peso_especifico=peso_especifico,
    sobrecarga_muerta=sobrecarga_muerta,
    amortiguamiento_ratio=(amortiguamiento / 100.0),
    modo_avanzado=modo_avanzado,
    b_col_cm=b_col_cm, h_col_cm=h_col_cm, b_viga_cm=b_viga_cm, h_viga_cm=h_viga_cm,
    A_col_cm2=A_col_cm2, I_col_cm4=I_col_cm4,
    A_viga_cm2=A_viga_cm2, I_viga_cm4=I_viga_cm4,
)

if st.session_state.get("param_estruct") != params_nuevos:
    st.session_state["param_estruct"] = params_nuevos
    st.session_state["A_col"] = A_col
    st.session_state["I_col"] = I_col
    st.session_state["A_viga"] = A_viga
    st.session_state["I_viga"] = I_viga

st.markdown("---")

# =============================================================================
# === SECCIÓN 2: MODELO + MASAS + TRANSFORMACIÓN + CONDENSACIÓN + PLOT ========
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st

from funciones_usuario import (
    b2_get_params_from_param_estruct,
    b2_params_key,
    b2_generar_modelo,
    plot_structure, 
    calcular_peso_total_estructura,
)

# -------------------------------------------------------------------------
# ✅ Texto
# -------------------------------------------------------------------------
T["en"].update({
    "b2_title": "Geometric and structural model definition",
    "b2_desc": (
        "Generates the frame geometry, assembles global K and computes "
        "M_cond and K_cond (rigid diaphragm + static condensation to 1 lateral DOF per floor)."
    ),

    "b2_ctrl": "Model control",
    "b2_btn": "Generate structural model",
    "b2_define_first": "First, define the parameters in Section 1.",
    "b2_success": "Structural model generated successfully.",
    "b2_summary": "Model summary",

    "b2_nodes": "Nodes",
    "b2_elems": "Elements",
    "b2_dofs": "DOF",

    "b2_mats": "Matrices",
    "b2_first_generate": "First generate the structural model.",

    "b2_Kglob": "View global stiffness matrix",
    "b2_Mcond": "View condensed mass matrix",
    "b2_T": "View transformation matrix",
    "b2_Kcond": "View condensed stiffness matrix",
    "b2_not_ready_M": "M_cond is not ready.",
    "b2_not_ready_K": "K_cond is not ready.",

    "b2_check": "Approximate lateral stiffness check",
    "b2_floor": "Story",
    "b2_k_model": "k_model",
    "b2_k_aprox": "k_aprox",
    "b2_ratio": "ratio",

    "b2_plot_title": "2D structural model",
    "b2_plot_wait": "The plot will appear here once you generate the model.",

    "b2_err": "Error while generating the model",
    "b2_warn_pinv": "Kss is singular; pseudo-inverse (pinv) was used.",
    "b2_err_no_floors": "There are no floors (y>0).",
    "b2_check_note": (
    "Approximate comparison between the stiffness obtained from the condensed model "
    "(k_model) and a simplified column-based estimate "
    "(k_aprox = n_col · 12EI/h³). "
    "Values close to 1 indicate good agreement. "
    "Intermediate stories usually show the best correlation, while the first and "
    "top stories may present larger differences due to global frame interaction "
    "and boundary effects."
    ),

    "b2_weight_breakdown": "Structural weight breakdown",
    "b2_weight_cols": "Columns weight",
    "b2_weight_beams": "Beams weight",
    "b2_weight_sdl": "Additional dead load weight",
    "b2_weight_total": "Total structural weight",
    "b2_mass_total": "Equivalent total mass",
    "b2_weight_note": (
        "This summary is computed from the complete structural elements. "
        "It is reported separately from the condensed mass matrix used in the reduced dynamic model."
    ),
})

T["es"].update({
    "b2_title": "Definición geométrica y estructural del modelo",
    "b2_desc": (
        "Genera la geometría del pórtico, ensambla K global y calcula "
        "M_cond y K_cond (diafragma rígido + condensación estática a 1 GDL lateral por piso)."
    ),

    "b2_ctrl": "Control del modelo",
    "b2_btn": "Generar modelo estructural",
    "b2_define_first": "Primero define los parámetros en la Sección 1.",
    "b2_success": "Modelo estructural generado correctamente.",
    "b2_summary": "Resumen del modelo",

    "b2_nodes": "Nodos",
    "b2_elems": "Elementos",
    "b2_dofs": "GDL",

    "b2_mats": "Matrices",
    "b2_first_generate": "Primero genera el modelo estructural.",

    "b2_Kglob": "Ver matriz global de rigidez",
    "b2_Mcond": "Ver matriz de masas condensada",
    "b2_T": "Ver matriz de transformación",
    "b2_Kcond": "Ver matriz de rigidez condensada",
    "b2_not_ready_M": "M_cond no está listo.",
    "b2_not_ready_K": "K_cond no está listo.",

    "b2_check": "Verificación aproximada de rigidez lateral por piso",
    "b2_floor": "Piso",
    "b2_k_model": "k_modelo",
    "b2_k_aprox": "k_aprox",
    "b2_ratio": "ratio",

    "b2_plot_title": "Modelo estructural 2D",
    "b2_plot_wait": "Aquí aparecerá el gráfico cuando generes el modelo.",

    "b2_err": "Error al generar el modelo",
    "b2_warn_pinv": "Kss singular; se usó pseudo-inversa (pinv).",
    "b2_err_no_floors": "No hay pisos (y>0).",
    "b2_check_note": (
    "Comparación aproximada entre la rigidez obtenida del modelo condensado "
    "(k_modelo) y una estimación simplificada basada únicamente en columnas "
    "(k_aprox = n_col · 12EI/h³). "
    "Valores cercanos a 1 indican buena concordancia. "
    "Los pisos intermedios suelen mostrar el mejor ajuste, mientras que el primer "
    "y último piso pueden presentar mayores diferencias debido a la interacción "
    "global del pórtico y a los efectos de borde del sistema estructural."
    ),

    "b2_weight_breakdown": "Desglose del peso estructural",
    "b2_weight_cols": "Peso de columnas",
    "b2_weight_beams": "Peso de vigas",
    "b2_weight_sdl": "Peso de sobrecarga muerta adicional",
    "b2_weight_total": "Peso total estructural",
    "b2_mass_total": "Masa total equivalente",
    "b2_weight_note": (
        "Este resumen se calcula a partir de los elementos estructurales completos. "
        "Se reporta por separado de la matriz de masas condensada usada en el modelo dinámico reducido."
    ),
})

# -------------------------------------------------------------------------
# Encabezado limpio y compacto
# -------------------------------------------------------------------------
st.markdown("""
<style>
h2 { margin-bottom: 0.15rem !important; }
p  { margin-top: 0rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"## 🏗️ {tr('b2_title')}")
st.caption(tr("b2_desc"))

# --- Layout principal ---
col_left, col_right = st.columns([0.50, 0.50], gap="large")

# -------------------------------------------------------------------------
# Helper: render de resumen (para que salga en el MISMO click)
# -------------------------------------------------------------------------
def _render_summary(where):
    if st.session_state.get("model_summary") is not None:
        sm = st.session_state["model_summary"]
        with where:
            st.markdown(f"#### {tr('b2_summary')}")
            r1, r2, r3 = st.columns(3)
            r1.metric(tr("b2_nodes"), sm.get("n_nodes", "—"))
            r2.metric(tr("b2_elems"), sm.get("n_elems", "—"))
            r3.metric(tr("b2_dofs"),  sm.get("n_dofs",  "—"))

# =========================
# IZQUIERDA: Control + Resumen
# =========================
with col_left:
    st.markdown(f"### {tr('b2_ctrl')}")

    msg_ph = st.empty()
    summary_ph = st.container()

    # ✅ Éxito persistente mientras el modelo esté listo
    if st.session_state.get("geom_ready", False) and st.session_state.get("model_summary") is not None:
        msg_ph.success(tr("b2_success"))

    _render_summary(summary_ph)

    hay_params = st.session_state.get("param_estruct") is not None
    generar = st.button(
        tr('b2_btn'),
        use_container_width=True,
        disabled=not hay_params
    )
    if not hay_params:
        st.info(tr("b2_define_first"))

# -------------------------------------------------------------------------
# Generación: solo si se presiona el botón
# -------------------------------------------------------------------------
if generar:
    try:
        p = b2_get_params_from_param_estruct(st.session_state.get("param_estruct", {}))
        pkey = b2_params_key(p)

        # sincronizar (si tu app lo usa luego)
        st.session_state["peso_especifico"]   = p["peso_especifico"]
        st.session_state["sobrecarga_muerta"] = p["sobrecarga_muerta"]
        st.session_state["amortiguamiento"]   = p["amortiguamiento"]
        st.session_state["E"]                 = p["E"]
        st.session_state["A_col"]             = p["A_col"]
        st.session_state["I_col"]             = p["I_col"]
        st.session_state["A_viga"]            = p["A_viga"]
        st.session_state["I_viga"]            = p["I_viga"]

        # motor (con warning callback a streamlit)
        out = b2_generar_modelo(
            p=p,
            Element=Element,
            generar_gdl_map=generar_gdl_map,
            assemble_global_stiffness=assemble_global_stiffness,
            calcular_matriz_masas_por_piso=calcular_matriz_masas_por_piso if "calcular_matriz_masas_por_piso" in globals() else None,
            warn_callback=lambda m: st.warning(tr("b2_warn_pinv")) if "pinv" in m.lower() else st.warning(m),
        )

        # Guardar session_state
        st.session_state["model_key"]          = pkey
        st.session_state["nodes"]              = out["nodes"]
        st.session_state["element_node_pairs"] = out["element_node_pairs"]
        st.session_state["propiedades"]        = out["propiedades"]
        st.session_state["gdl_map"]            = out["gdl_map"]
        st.session_state["elements"]           = out["elements"]
        st.session_state["K_global"]           = out["K_global"]
        st.session_state["M_cond"]             = out["M_cond"]
        st.session_state["T_trans"]            = out["T_trans"]
        st.session_state["K_cond"]             = out["K_cond"]
        st.session_state["k_modelo"]           = out["k_modelo"]
        st.session_state["k_aprox"]            = out["k_aprox"]
        st.session_state["ratio_k"]            = out["ratio_k"]
        st.session_state["model_summary"]      = out["model_summary"]

        # -------------------------------------------------------------
        # Peso total real de la estructura
        # -------------------------------------------------------------
        resumen_peso = calcular_peso_total_estructura(
            nodes=out["nodes"],
            element_node_pairs=out["element_node_pairs"],
            propiedades=out["propiedades"],
            peso_especifico=p["peso_especifico"],
            sobrecarga_muerta=p["sobrecarga_muerta"],
            b_col_x=p.get("b_col_x", 0.50),
        )

        st.session_state["peso_columnas"] = resumen_peso["peso_columnas"]
        st.session_state["peso_vigas"] = resumen_peso["peso_vigas"]
        st.session_state["peso_sobrecarga_muerta_total"] = resumen_peso["peso_sobrecarga_muerta"]
        st.session_state["peso_total_estructura"] = resumen_peso["peso_total"]
        st.session_state["masa_total_estructura"] = resumen_peso["masa_total"]
        
        st.session_state["geom_ready"] = True
        st.rerun()

    except Exception as e:
        st.session_state["geom_ready"] = False

        st.session_state["peso_columnas"] = None
        st.session_state["peso_vigas"] = None
        st.session_state["peso_sobrecarga_muerta_total"] = None
        st.session_state["peso_total_estructura"] = None
        st.session_state["masa_total_estructura"] = None

        st.error(f"{tr('b2_err')}: {e}")

# -------------------------------------------------------------------------
# Mostrar resultados persistentes
# -------------------------------------------------------------------------
requeridos = ["nodes", "element_node_pairs", "propiedades", "gdl_map", "elements", "K_global"]
faltantes = [k for k in requeridos if k not in st.session_state or st.session_state[k] is None]

with col_left:
    st.markdown(f"### {tr('b2_mats')}")

    if faltantes:
        st.info(tr("b2_first_generate"))
    else:
        nodes              = st.session_state["nodes"]
        element_node_pairs = st.session_state["element_node_pairs"]
        propiedades        = st.session_state["propiedades"]
        gdl_map            = st.session_state["gdl_map"]
        elements           = st.session_state["elements"]
        K_global           = st.session_state["K_global"]

        M_cond  = st.session_state.get("M_cond")
        K_cond  = st.session_state.get("K_cond")

        with st.expander(tr("b2_Kglob"), expanded=False):
            st.write(f"Dimensión: {K_global.shape}")
            st.dataframe(np.round(K_global, 3), height=260, use_container_width=True)

        if M_cond is not None:
            with st.expander(tr("b2_Mcond"), expanded=False):
                st.write(f"Dimensión: {np.array(M_cond).shape}")
                st.dataframe(np.round(M_cond, 5), use_container_width=True)
        else:
            st.warning(tr("b2_not_ready_M"))

        if K_cond is not None:
            with st.expander(tr("b2_Kcond"), expanded=False):
                st.write(f"Dimensión: {K_cond.shape}")
                st.dataframe(np.round(K_cond, 3), use_container_width=True)
        else:
            st.warning(tr("b2_not_ready_K"))

        k_modelo = st.session_state.get("k_modelo")
        k_aprox  = st.session_state.get("k_aprox")
        ratio_k  = st.session_state.get("ratio_k")
        peso_columnas = st.session_state.get("peso_columnas")
        peso_vigas = st.session_state.get("peso_vigas")
        peso_sobrecarga_muerta_total = st.session_state.get("peso_sobrecarga_muerta_total")
        peso_total_estructura = st.session_state.get("peso_total_estructura")
        masa_total_estructura = st.session_state.get("masa_total_estructura")    

        if (k_modelo is not None) and (k_aprox is not None) and (ratio_k is not None):

            with st.expander(tr("b2_check"), expanded=False):
        
                st.caption(tr("b2_check_note"))
        
                df_check = pd.DataFrame({
                    tr("b2_floor"): np.arange(1, len(k_modelo) + 1),
                    tr("b2_k_model"): np.round(k_modelo, 6),
                    tr("b2_k_aprox"): np.round(k_aprox, 6),
                    tr("b2_ratio"): np.round(ratio_k, 3),
                })
        
                st.dataframe(df_check, use_container_width=True)

        if (
            (peso_columnas is not None) and
            (peso_vigas is not None) and
            (peso_sobrecarga_muerta_total is not None) and
            (peso_total_estructura is not None) and
            (masa_total_estructura is not None)
        ):
            with st.expander(tr("b2_weight_breakdown"), expanded=False):

                st.caption(tr("b2_weight_note"))

                lang_now = st.session_state.get("lang", "en")
                col_name = "Concepto" if lang_now == "es" else "Concept"
                val_name = "Valor" if lang_now == "es" else "Value"
                unit_name = "Unidades" if lang_now == "es" else "Units"

                df_pesos = pd.DataFrame({
                    col_name: [
                        tr("b2_weight_cols"),
                        tr("b2_weight_beams"),
                        tr("b2_weight_sdl"),
                        tr("b2_weight_total"),
                        tr("b2_mass_total"),
                    ],
                    val_name: [
                        round(peso_columnas, 6),
                        round(peso_vigas, 6),
                        round(peso_sobrecarga_muerta_total, 6),
                        round(peso_total_estructura, 6),
                        round(masa_total_estructura, 6),
                    ],
                    unit_name: [
                        "Tf",
                        "Tf",
                        "Tf",
                        "Tf",
                        "Tf·s²/m",
                    ],
                })

                st.dataframe(df_pesos, use_container_width=True)

# -------------------------------------------------------------------------
# Derecha: gráfico
# -------------------------------------------------------------------------
with col_right:
    st.markdown(f"### {tr('b2_plot_title')}")

    if faltantes:
        st.info(tr("b2_plot_wait"))
    else:
        nodes        = st.session_state["nodes"]
        elements     = st.session_state["elements"]
        gdl_map      = st.session_state["gdl_map"]
        propiedades  = st.session_state["propiedades"]

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
            propiedades=propiedades,
            title=tr("b2_plot_title")
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

        st.markdown(
            """
            <style>
            iframe[title="matplotlib.figure.Figure"] {
                margin-bottom: -35px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# =============================================================================
# ========== BLOQUE 3: NEC-24 (izq) + REGISTRO AT2 PEER (der) =================
# =============================================================================
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from funciones_usuario import (
    nec24_espectro,
    leer_archivo_bytes_a_texto,
    procesar_registro,
    G_STD,
    detectar_formato_y_extraer,
)

# -------------------------------------------------------------------------
# ✅ Textos EN/ES (solo para este bloque)
# -------------------------------------------------------------------------
T["en"].update({
    "b3_title": "NEC-24 + Seismic record",

    "b3_nec_params": "NEC-24 spectrum parameters",
    "b3_need_model_nec": "First generate the **structural model** in Section 2 to enable NEC-24.",
    "b3_z": "Seismic intensity (z)",
    "b3_zone": "Seismic zone",
    "b3_soil": "Soil type",
    "b3_Ie": "Importance factor (Ie)",
    "b3_risk_cat": "Risk category",

    "h_b3_z": "Design seismic intensity parameter z (NEC-24).",
    "h_b3_zone": "Seismic zoning used by NEC-24 (I to V).",
    "h_b3_soil": "Site soil class (A to E) used to compute Fa, Fd, Fs.",
    "h_b3_Ie": "Importance factor Ie associated with the selected risk category.",
    "h_b3_risk_cat": "Defines importance factor Ie according to NEC-24.",

    "b3_nec_spec": "Response spectrum – NEC-24",
    "b3_nec_wait": "The NEC-24 spectrum will be shown once the model is ready.",
    "b3_T": "Period T [s]",
    "b3_Sa": "Sa [g]",
    "b3_placeholder": "(placeholder)",
    "b3_elastic": "Elastic",
    "b3_design_spec": "Design spectrum",
    "b3_sds_sd1": "SDS = {SDS:.3f} g, SD1 = {SD1:.3f} g·s, Ie = {Ie:.2f}",
    "b3_coeffs": "Coefficients: Fa={Fa:.2f}, Fd={Fd:.2f}, Fs={Fs:.2f}",
    "b3_nec_dl_btn": "Download NEC-24 Excel",
    "b3_nec_dl_help": "Exports 2 columns: period, design spectrum.",

    "b3_region": "Region of Ecuador",
    "b3_region_costa": "Coast",
    "b3_region_sierra": "Highlands and Amazon",
    "h_b3_region": "Defines exponent r for the NEC-24 descending branch.",

    "b3_rec_load": "AT2 seismic record (PEER)",
    "b3_file": "Select a PEER .at2 file",
    "b3_file_help": "Only .at2 files already prepared by PEER are allowed in this simplified workflow.",
    "b3_proc": "Apply filtering + baseline correction",
    "h_b3_proc": "Applies: linear detrend + Butterworth bandpass + baseline correction (v and u).",
    "b3_only_at2": "Only .at2 files are accepted in this section.",
    "b3_peer_note": "Use a PEER AT2 record already selected and prepared by the user.",
    "b3_need_model_rec": "First generate the **structural model** (Section 2) to enable the record.",
    "b3_no_file": "Upload a PEER .at2 file to continue.",

    "b3_event": "Event",
    "b3_dt": "Time step",
    "b3_dur": "Total duration",
    "b3_npts": "Number of points",
    "b3_units_in": "Input units",
    "b3_reg_title": "Seismic record – {name}",
    "b3_acc": "Acceleration [m/s²]",
    "b3_vel": "Velocity [m/s]",
    "b3_disp": "Displacement [m]",
    "b3_time": "Time [s]",
    "b3_orig": "Original",
    "b3_proc_lab": "Filtered + corrected",

    "b3_dl_hdr": "Download record (Excel)",
    "b3_dl_pick": "Choose data to export",
    "b3_dl_btn": "Download Excel (.xlsx)",
    "h_b3_dl_pick": "Exports 4 columns: time, acceleration, velocity, displacement.",
    "b3_dl_opt_orig": "Original",
    "b3_dl_opt_proc": "Filtered + baseline-corrected",
    "b3_dl_opt_final": "Final used in analysis",
})

T["es"].update({
    "b3_title": "NEC-24 + Registro sísmico",

    "b3_nec_params": "Parámetros del espectro NEC-24",
    "b3_need_model_nec": "⚙️ Primero genera el **modelo estructural** en la Sección 2 para habilitar NEC-24.",
    "b3_z": "Intensidad sísmica (z)",
    "b3_zone": "Zona sísmica",
    "b3_soil": "Tipo de suelo",
    "b3_Ie": "Factor de importancia (Ie)",
    "b3_risk_cat": "Categoría de riesgo",

    "h_b3_z": "Parámetro de intensidad sísmica z (NEC-24).",
    "h_b3_zone": "Zonificación sísmica usada por la NEC-24 (I a V).",
    "h_b3_soil": "Clase de suelo (A a E) para calcular Fa, Fd, Fs.",
    "h_b3_Ie": "Factor de importancia Ie asociado a la categoría de riesgo seleccionada.",
    "h_b3_risk_cat": "Define el factor de importancia Ie según la NEC-24.",

    "b3_nec_spec": "Espectro de respuesta – NEC-24",
    "b3_nec_wait": "📌 El espectro NEC-24 se mostrará cuando el modelo esté listo.",
    "b3_T": "Período T [s]",
    "b3_Sa": "Sa [g]",
    "b3_placeholder": "(placeholder)",
    "b3_elastic": "Elástico",
    "b3_design_spec": "Espectro de diseño",
    "b3_sds_sd1": "SDS = {SDS:.3f} g, SD1 = {SD1:.3f} g·s, Ie = {Ie:.2f}",
    "b3_coeffs": "Coeficientes: Fa={Fa:.2f}, Fd={Fd:.2f}, Fs={Fs:.2f}",
    "b3_nec_dl_btn": "Descargar NEC-24 Excel",
    "b3_nec_dl_help": "Exporta 2 columnas: periodo, espectro de diseño.",

    "b3_region": "Región del Ecuador",
    "b3_region_costa": "Costa",
    "b3_region_sierra": "Sierra y oriente",
    "h_b3_region": "Define el exponente r para la rama descendente del espectro NEC-24.",

    "b3_rec_load": "Registro sísmico AT2 (PEER)",
    "b3_file": "📁 Selecciona un archivo PEER .at2",
    "b3_file_help": "En este flujo simplificado solo se aceptan archivos .at2 ya preparados por PEER.",
    "b3_proc": "Aplicar filtrado + corrección de línea base",
    "h_b3_proc": "Aplica: detrend lineal + filtro Butterworth pasa banda + corrección de línea base (v y u).",
    "b3_only_at2": "En esta sección solo se aceptan archivos .at2.",
    "b3_peer_note": "Use un registro AT2 de PEER ya seleccionado y preparado por el usuario.",
    "b3_need_model_rec": "⚙️ Primero genera el **modelo estructural** (Sección 2) para habilitar el registro.",
    "b3_no_file": "Cargue un archivo PEER .at2 para continuar.",

    "b3_event": "Evento",
    "b3_dt": "Paso de tiempo",
    "b3_dur": "Duración total",
    "b3_npts": "Número de puntos",
    "b3_units_in": "Unidades de entrada",
    "b3_reg_title": "Registro sísmico – {name}",
    "b3_acc": "Aceleración [m/s²]",
    "b3_vel": "Velocidad [m/s]",
    "b3_disp": "Desplazamiento [m]",
    "b3_time": "Tiempo [s]",
    "b3_orig": "Original",
    "b3_proc_lab": "Filtrado + corregido",

    "b3_dl_hdr": "Descargar registro (Excel)",
    "b3_dl_pick": "Elegir datos a exportar",
    "b3_dl_btn": "Descargar Excel (.xlsx)",
    "h_b3_dl_pick": "Exporta 4 columnas: tiempo, aceleracion, velocidad, desplazamiento.",
    "b3_dl_opt_orig": "Original",
    "b3_dl_opt_proc": "Filtrado + corregido (línea base)",
    "b3_dl_opt_final": "Final usado en el análisis",
})

# -------------------------------------------------------------------------
# ✅ Helper: Excel NEC-24
# -------------------------------------------------------------------------
def build_nec24_excel_bytes(T_spec, Sa_design):
    df_nec = pd.DataFrame({
        "periodo": np.asarray(T_spec, dtype=float).ravel(),
        "diseno": np.asarray(Sa_design, dtype=float).ravel(),
    })

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df_nec.to_excel(writer, index=False, sheet_name="NEC24")
    bio.seek(0)
    return bio.getvalue()

# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
st.markdown(f"## 📌 {tr('b3_title')}")

BG         = "#2B3141"
COLOR_TEXT = "#E8EDF2"
COLOR_GRID = "#5B657A"

geom_ok = bool(st.session_state.get("geom_ready", False))

# -------------------------------------------------------------------------
# ✅ Ajustes visuales
# -------------------------------------------------------------------------
NEC24_PAD_PX = 15
NEC24_FIG_H  = 3.25

st.markdown(f"""
<style>
.nec24-equalizer {{
  height: {NEC24_PAD_PX}px;
}}

.compact-download div[data-testid="stVerticalBlock"]{{ gap:0.25rem; }}
.compact-download [data-testid="stSelectbox"]{{ padding-top:0rem !important; padding-bottom:0rem !important; }}
.compact-download [data-testid="stDownloadButton"]{{ padding-top:0rem !important; padding-bottom:0rem !important; }}
.compact-download button{{ padding-top:0.25rem !important; padding-bottom:0.25rem !important; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Layout principal (2 columnas)
# =============================================================================
col_left, col_right = st.columns([1.05, 1.95], gap="large")

# =============================================================================
# IZQUIERDA: NEC-24
# =============================================================================
with col_left:
    with st.container(border=True):
        st.markdown(f"### 🧩 {tr('b3_nec_params')}")

        if not geom_ok:
            st.info(tr("b3_need_model_nec"))
            z = 0.47
            zona_sismica = "IV"
            tipo_suelo = "C"
            categoria = "II"
            Ie = 1.0
            region_ecuador = tr("b3_region_sierra")
            r_nec = 1.0
        else:
            c1, c2 = st.columns(2)

            with c1:
                z = st.number_input(
                    tr("b3_z"), 0.1, 1.0, 0.47, 0.01,
                    key="nec_z", help=tr("h_b3_z")
                )
                zona_sismica = st.selectbox(
                    tr("b3_zone"), ["I", "II", "III", "IV", "V"],
                    index=3, key="nec_zona", help=tr("h_b3_zone")
                )
                tipo_suelo = st.selectbox(
                    tr("b3_soil"), ["A", "B", "C", "D", "E"],
                    index=2, key="nec_suelo", help=tr("h_b3_soil")
                )

            with c2:
                categoria = st.selectbox(
                    tr("b3_risk_cat"),
                    ["I", "II", "III", "IV"],
                    index=1,
                    key="nec_risk_cat",
                    help=tr("h_b3_risk_cat")
                )

                IE_MAP = {
                    "I": 1.00,
                    "II": 1.00,
                    "III": 1.25,
                    "IV": 1.50,
                }
                Ie = IE_MAP[categoria]

                region_ecuador = st.selectbox(
                    tr("b3_region"),
                    [tr("b3_region_costa"), tr("b3_region_sierra")],
                    index=1,
                    key="nec_region",
                    help=tr("h_b3_region")
                )

                r_nec = 1.2 if region_ecuador == tr("b3_region_costa") else 1.0

            st.session_state["nec24_params"] = {
                "z": float(z),
                "zona": str(zona_sismica),
                "suelo": str(tipo_suelo),
                "Ie": float(Ie),
                "categoria": str(categoria),
                "region": str(region_ecuador),
                "r": float(r_nec),
            }

    st.write("")

    with st.container(border=True):
        st.markdown(f"### 📈 {tr('b3_nec_spec')}")

        if not geom_ok:
            st.info(tr("b3_nec_wait"))
            T_spec = np.linspace(0.0, 5.0, 120)
            Sa_elast = np.zeros_like(T_spec)
            SDS = SD1 = Fa = Fd = Fs = 0.0

            fig, ax = plt.subplots(figsize=(6.0, NEC24_FIG_H))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            ax.plot(T_spec, Sa_elast, lw=0.75, alpha=0.5, label=tr("b3_placeholder"))
            ax.set_xlabel(tr("b3_T"), color=COLOR_TEXT)
            ax.set_ylabel(tr("b3_Sa"), color=COLOR_TEXT)
            ax.tick_params(colors=COLOR_TEXT)
            ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig, use_container_width=True)

            st.markdown('<div class="nec24-equalizer"></div>', unsafe_allow_html=True)

        else:
            T_spec, Sa_elast, _Sa_inelas_unused, SDS, SD1, Fa, Fd, Fs = nec24_espectro(
                z=float(z),
                zona=str(zona_sismica),
                suelo=str(tipo_suelo),
                R=1.0,
                r=float(r_nec),
                T_final=5.0,
                delta_t=0.01
            )

            Sa_design_plot = np.asarray(Sa_elast, dtype=float).ravel()

            st.session_state["SDS"] = float(SDS)
            st.session_state["SD1"] = float(SD1)
            st.session_state["r_nec"] = float(r_nec)
            st.session_state["rs_T_spec"] = np.asarray(T_spec, dtype=float).ravel()
            st.session_state["rs_Sa_design"] = np.asarray(Sa_design_plot, dtype=float).ravel()
            st.session_state["rs_Ie"] = float(Ie)

            st.caption(tr("b3_sds_sd1").format(SDS=SDS, SD1=SD1, Ie=Ie))

            colA, colB = st.columns([1.3, 1.0])

            with colA:
                st.caption(tr("b3_coeffs").format(Fa=Fa, Fd=Fd, Fs=Fs))

            with colB:
                nec24_xlsx_bytes = build_nec24_excel_bytes(T_spec, Sa_design_plot)
                st.download_button(
                    label=tr("b3_nec_dl_btn"),
                    data=nec24_xlsx_bytes,
                    file_name="NEC24_spectrum.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="b3_nec24_dl_btn",
                    help=tr("b3_nec_dl_help"),
                    use_container_width=True,
                )

            fig, ax = plt.subplots(figsize=(6.0, NEC24_FIG_H))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            ax.plot(T_spec, Sa_design_plot, lw=1.0, label=tr("b3_design_spec"))
            ax.set_xlabel(tr("b3_T"), color=COLOR_TEXT)
            ax.set_ylabel(tr("b3_Sa"), color=COLOR_TEXT)
            ax.tick_params(colors=COLOR_TEXT)
            ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            leg = ax.legend(framealpha=0.95)
            leg.get_frame().set_facecolor(BG)
            leg.get_frame().set_edgecolor(COLOR_GRID)
            for t in leg.get_texts():
                t.set_color(COLOR_TEXT)

            st.pyplot(fig, use_container_width=True)
            st.markdown('<div class="nec24-equalizer"></div>', unsafe_allow_html=True)

# =============================================================================
# DERECHA: REGISTRO AT2
# =============================================================================
with col_right:
    with st.container(border=True):
        st.markdown(f"### 〰️ {tr('b3_rec_load')}")

        col_ctrl, col_graf = st.columns([1.15, 2.55], gap="large")

        nombre = None
        unidad = None
        dt = None
        t_ag = None
        ag_orig = None
        vel_orig = None
        disp_orig = None
        ag_proc = None
        vel_proc = None
        disp_proc = None
        ag_base = None
        proc_disponible = False

        with col_ctrl:
            st.caption(tr("b3_peer_note"))

            uploaded = st.file_uploader(
                tr("b3_file"),
                type=["at2"],
                disabled=not geom_ok,
                help=tr("b3_file_help"),
                key="b3_at2_file"
            )

            aplicar_proc = st.checkbox(
                tr("b3_proc"),
                value=False,
                disabled=not geom_ok,
                help=tr("h_b3_proc"),
                key="b3_proc_check"
            )

            if not geom_ok:
                st.info(tr("b3_need_model_rec"))
            elif uploaded is None:
                st.info(tr("b3_no_file"))

        if geom_ok and uploaded is not None:
            if not uploaded.name.lower().endswith(".at2"):
                st.error(tr("b3_only_at2"))
                st.stop()

            try:
                raw = uploaded.read()
                texto = leer_archivo_bytes_a_texto(raw)

                nombre, unidad, dt, ag = detectar_formato_y_extraer(texto)
                ag = np.asarray(ag, dtype=float).ravel()
                dt = float(dt)

                if unidad == "cm/s²":
                    ag_orig = ag / 100.0
                elif unidad == "g":
                    ag_orig = ag * G_STD
                elif unidad == "m/s²":
                    ag_orig = ag
                else:
                    st.error(f"Unidad no reconocida: {unidad}")
                    st.stop()

            except Exception as e:
                st.error(f"Error al leer el archivo AT2: {e}")
                st.stop()

            out = procesar_registro(ag_orig, dt, aplicar_proc=aplicar_proc)

            t_ag = np.asarray(out["t"], dtype=float).ravel()
            vel_orig = np.asarray(out["vel_orig"], dtype=float).ravel()
            disp_orig = np.asarray(out["disp_orig"], dtype=float).ravel()
            ag_proc = out["ag_proc"]
            vel_proc = out["vel_proc"]
            disp_proc = out["disp_proc"]
            ag_base = np.asarray(out["ag_base"], dtype=float).ravel()

            if ag_proc is not None:
                ag_proc = np.asarray(ag_proc, dtype=float).ravel()
            if vel_proc is not None:
                vel_proc = np.asarray(vel_proc, dtype=float).ravel()
            if disp_proc is not None:
                disp_proc = np.asarray(disp_proc, dtype=float).ravel()

            proc_disponible = bool(
                aplicar_proc and
                (ag_proc is not None) and
                (vel_proc is not None) and
                (disp_proc is not None)
            )

            with col_ctrl:
                st.markdown(f"**{tr('b3_event')}:** {nombre}")
                st.markdown(f"**{tr('b3_units_in')}:** {unidad}")
                st.markdown(f"**{tr('b3_dt')}:** {dt:.4f} s")
                st.markdown(f"**{tr('b3_dur')}:** {t_ag[-1]:.2f} s")
                st.markdown(f"**{tr('b3_npts')}:** {len(ag_orig)}")

            COLOR_ORIG = "#9DBEF7"
            COLOR_PROC = "#FFD479"
            LW_ORIG_SOLO = 0.55
            LW_ORIG_OVER = 0.28
            LW_PROC = 0.28

            with col_graf:
                fig, axs = plt.subplots(3, 1, figsize=(9, 11.2), sharex=True)
                fig.patch.set_facecolor(BG)

                for ax in axs:
                    ax.set_facecolor(BG)
                    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
                    ax.tick_params(colors=COLOR_TEXT)
                    for s in ("top", "right"):
                        ax.spines[s].set_visible(False)

                axs[0].plot(
                    t_ag, ag_orig,
                    lw=(LW_ORIG_OVER if proc_disponible else LW_ORIG_SOLO),
                    color=COLOR_ORIG,
                    label=tr("b3_orig")
                )
                if proc_disponible:
                    axs[0].plot(
                        t_ag, ag_proc,
                        lw=LW_PROC, color=COLOR_PROC, label=tr("b3_proc_lab")
                    )
                axs[0].set_ylabel(tr("b3_acc"), color=COLOR_TEXT)
                axs[0].set_title(tr("b3_reg_title").format(name=nombre), color=COLOR_TEXT)

                axs[1].plot(
                    t_ag, vel_orig,
                    lw=(LW_ORIG_OVER if proc_disponible else LW_ORIG_SOLO),
                    color=COLOR_ORIG
                )
                if proc_disponible:
                    axs[1].plot(t_ag, vel_proc, lw=LW_PROC, color=COLOR_PROC)
                axs[1].set_ylabel(tr("b3_vel"), color=COLOR_TEXT)

                axs[2].plot(
                    t_ag, disp_orig,
                    lw=(LW_ORIG_OVER if proc_disponible else LW_ORIG_SOLO),
                    color=COLOR_ORIG
                )
                if proc_disponible:
                    axs[2].plot(t_ag, disp_proc, lw=LW_PROC, color=COLOR_PROC)
                axs[2].set_ylabel(tr("b3_disp"), color=COLOR_TEXT)
                axs[2].set_xlabel(tr("b3_time"), color=COLOR_TEXT)

                if proc_disponible:
                    leg0 = axs[0].legend(framealpha=0.85)
                    leg0.get_frame().set_facecolor(BG)
                    leg0.get_frame().set_edgecolor(COLOR_GRID)
                    for tt in leg0.get_texts():
                        tt.set_color(COLOR_TEXT)

                st.pyplot(fig, use_container_width=True)

            # -------------------------------------------------------------
            # ✅ Guardado principal en session_state
            # -------------------------------------------------------------
            st.session_state["rs_ready"] = True
            st.session_state["rs_nombre"] = str(nombre)
            st.session_state["rs_dt"] = float(dt)
            st.session_state["rs_t"] = np.asarray(t_ag, dtype=float).ravel()

            st.session_state["rs_ag_orig"] = np.asarray(ag_orig, dtype=float).ravel()
            st.session_state["rs_vel_orig"] = np.asarray(vel_orig, dtype=float).ravel()
            st.session_state["rs_disp_orig"] = np.asarray(disp_orig, dtype=float).ravel()

            st.session_state["rs_ag_base"] = np.asarray(ag_base, dtype=float).ravel()

            st.session_state["rs_ag_proc"] = np.asarray(ag_proc, dtype=float).ravel() if proc_disponible else None
            st.session_state["rs_vel_proc"] = np.asarray(vel_proc, dtype=float).ravel() if proc_disponible else None
            st.session_state["rs_disp_proc"] = np.asarray(disp_proc, dtype=float).ravel() if proc_disponible else None
            st.session_state["rs_proc_on"] = bool(proc_disponible)

            # -------------------------------------------------------------
            # ✅ Registro final usado en el análisis
            #    Si hay procesamiento activo, usa la señal procesada.
            #    Si no, usa la original convertida a m/s².
            # -------------------------------------------------------------
            ag_final = np.asarray(ag_base, dtype=float).ravel()

            out_final = procesar_registro(
                ag_final,
                float(dt),
                aplicar_proc=False
            )
            vel_final = np.asarray(out_final["vel_orig"], dtype=float).ravel()
            disp_final = np.asarray(out_final["disp_orig"], dtype=float).ravel()

            st.session_state["ag_filt"] = np.asarray(ag_final, dtype=float).ravel()
            st.session_state["dt"] = float(dt)
            st.session_state["t_ag"] = np.asarray(t_ag, dtype=float).ravel()

            st.session_state["rs_ag_final"] = np.asarray(ag_final, dtype=float).ravel()
            st.session_state["rs_vel_final"] = np.asarray(vel_final, dtype=float).ravel()
            st.session_state["rs_disp_final"] = np.asarray(disp_final, dtype=float).ravel()

# =============================================================================
# DESCARGA FINAL DEL REGISTRO (3 OPCIONES)
# =============================================================================
rs_ready = bool(st.session_state.get("rs_ready", False))

if geom_ok and rs_ready and ("rs_t" in st.session_state):
    with col_right:
        with col_ctrl:
            st.markdown('<div class="compact-download">', unsafe_allow_html=True)
            st.caption(f"📥 **{tr('b3_dl_hdr')}**")

            opts = [tr("b3_dl_opt_orig")]
            if bool(st.session_state.get("rs_proc_on", False)):
                opts.append(tr("b3_dl_opt_proc"))
            opts.append(tr("b3_dl_opt_final"))

            pick = st.selectbox(
                tr("b3_dl_pick"),
                options=opts,
                index=(len(opts) - 1),
                key="b3_dl_pick_opt",
                help=tr("h_b3_dl_pick"),
                label_visibility="collapsed",
            )

            t_exp = np.asarray(st.session_state["rs_t"], dtype=float).ravel()

            if pick == tr("b3_dl_opt_orig"):
                a_exp = np.asarray(st.session_state["rs_ag_orig"], dtype=float).ravel()
                v_exp = np.asarray(st.session_state["rs_vel_orig"], dtype=float).ravel()
                u_exp = np.asarray(st.session_state["rs_disp_orig"], dtype=float).ravel()
                tag = "orig"

            elif pick == tr("b3_dl_opt_proc") and bool(st.session_state.get("rs_proc_on", False)):
                a_exp = np.asarray(st.session_state["rs_ag_proc"], dtype=float).ravel()
                v_exp = np.asarray(st.session_state["rs_vel_proc"], dtype=float).ravel()
                u_exp = np.asarray(st.session_state["rs_disp_proc"], dtype=float).ravel()
                tag = "proc"

            else:
                a_exp = np.asarray(st.session_state["rs_ag_final"], dtype=float).ravel()
                v_exp = np.asarray(st.session_state["rs_vel_final"], dtype=float).ravel()
                u_exp = np.asarray(st.session_state["rs_disp_final"], dtype=float).ravel()
                tag = "final"

            df_xlsx = pd.DataFrame({
                "tiempo": t_exp,
                "aceleracion": a_exp,
                "velocidad": v_exp,
                "desplazamiento": u_exp,
            })

            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                df_xlsx.to_excel(writer, index=False, sheet_name="registro")
            bio.seek(0)

            safe_name = "".join([
                c if (c.isalnum() or c in ("_", "-", ".")) else "_"
                for c in str(st.session_state.get("rs_nombre", "registro"))
            ])
            file_name = f"registro_{safe_name}_{tag}.xlsx"

            st.download_button(
                label=tr("b3_dl_btn"),
                data=bio.getvalue(),
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="b3_dl_btn_xlsx",
                use_container_width=True,
            )

            st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# == BLOQUE 4: DISEÑO DEL AISLADOR LRB (MODAL + RAYLEIGH + DISEÑO + GRÁFICO) ==
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
from numpy.linalg import inv, eig
import math
from funciones_usuario import _km_key, diseno_aislador_LRB, set_style_arctic_dark, plot_ciclo_histeretico_lrb

# -------------------------------------------------------------------------
# ✅ Textos EN/ES (solo para este bloque) + HELPERS
# -------------------------------------------------------------------------
T["en"].update({
    "b4_title": "LRB isolator design",
    "b4_need_prev": "First generate the structural model and condensation (Section 2).",

    "b4_subtitle": "Design parameters",
    "b4_zeta_note": "Rayleigh modal damping is set by default to **ζ = 5%**.",

    "b4_modal_data": "Modal data",
    "h_b4_modal_data": "Shows natural frequencies and periods from the condensed model.",

    "b4_design_mode_hdr": "Design mode",
    "b4_design_mode": "Design mode",
    "h_b4_design_mode": "Choose automatic design or set a target isolation period.",

    "b4_mode_auto": "Automatic",
    "b4_mode_Tobj": "By target period",

    "b4_Tobj": "T_target [s]",
    "h_b4_Tobj": "Target isolation period used only when 'By target period' is selected.",

    "b4_run": "Run isolator design",
    "h_b4_run": "Runs the LRB design using NEC spectrum parameters and the current condensed model.",

    "b4_missing_prev": "Missing previous data: {keys}. Run the NEC-24 spectrum block before design.",
    "b4_missing_res": "Missing keys in resultados_ais: {keys}",

    "b4_ok": "Isolator design completed successfully.",
    "b4_err": "Error during isolator design: {e}",

    "b4_need_run_plot": "Run the design to see the isolator plot.",
    "b4_plot_hdr": "Isolator force–displacement characteristic curve",
    "b4_plot_sub": "(Bilinear model – Units: Tonf and m)",

    # ✅ Box
    "b4_box_hdr": "=== INDIVIDUAL LRB ISOLATOR PROPERTIES ===",
    "b4_box_keff": "Effective stiffness 𝐊ₑ𝑓𝑓",
    "b4_box_ceq":  "Viscous damping 𝐂ₑ𝑞",
    "b4_box_ke":   "Initial elastic stiffness 𝐊ₑ",
    "b4_box_kp":   "Post-yield stiffness 𝐊ₚ",
    "b4_box_fy":   "Yield force 𝐅ᵧ",
    "b4_box_r":    "Stiffness ratio 𝐫 = 𝐊ₚ/𝐊ₑ",

    # ✅ Hysteresis texts
    "b4_hyst_title": "Bilinear hysteretic cycle – LRB isolator",
    "b4_hyst_xlabel": "Displacement Δ (m)",
    "b4_hyst_ylabel": "Shear force F (Tonf)",

    # ✅ Modal table headers
    "b4_mode": "Mode",
    "b4_f": "f [Hz]",
    "b4_T": "T [s]",
    
    # ✅ Warning text prefix
    "b4_warn_hdr": "Period objective warning",

    "b4_box_Tm": "Effective period 𝐓ₘ",
    "b4_box_betaM": "Effective damping βₘ",
    "b4_box_DM": "Maximum displacement 𝐃ₘ",
    "b4_box_dy": "Yield displacement δᵧ",
    "b4_box_DL": "Lead core diameter 𝐃ₗ",
    "b4_box_DB": "Rubber diameter 𝐃ᵦ",
    "b4_box_tr": "Total rubber thickness 𝐭ᵣ",

    "b4_box_chk_dy": "δᵧ < Dₘ",
    "b4_box_chk_k": "Kₑ > Kₚ",
    "b4_box_yes": "Yes",
    "b4_box_no": "No",
})

T["es"].update({
    "b4_title": "Diseño del aislador LRB",
    "b4_need_prev": "⚙️ Primero genera el modelo estructural y la condensación (Sección 2).",

    "b4_subtitle": "Parámetros de diseño",
    "b4_zeta_note": "El amortiguamiento modal de Rayleigh se fija por defecto en **ζ = 5%**.",

    "b4_modal_data": "📌 Datos modales",
    "h_b4_modal_data": "Muestra frecuencias y períodos naturales del modelo condensado.",

    "b4_design_mode_hdr": "Modo de diseño",
    "b4_design_mode": "Modo de diseño",
    "h_b4_design_mode": "Elige diseño automático o define un período objetivo de aislamiento.",

    "b4_mode_auto": "Automático",
    "b4_mode_Tobj": "Por período objetivo",

    "b4_Tobj": "T_objetivo [s]",
    "h_b4_Tobj": "Período objetivo del aislado (solo se usa en 'Por período objetivo').",

    "b4_run": "Ejecutar diseño del aislador",
    "h_b4_run": "Ejecuta el diseño LRB usando NEC-24 y el modelo condensado actual.",

    "b4_missing_prev": "⚠️ Faltan datos previos: {keys}. Ejecuta el espectro NEC-24 antes del diseño.",
    "b4_missing_res": "❌ Faltan en resultados_ais: {keys}",

    "b4_ok": "✅ Diseño del aislador completado correctamente.",
    "b4_err": "❌ Error durante el diseño del aislador: {e}",

    "b4_need_run_plot": "ℹ️ Ejecuta el diseño para ver el gráfico del aislador.",
    "b4_plot_hdr": "Curva característica Fuerza–Desplazamiento del Aislador LRB",
    "b4_plot_sub": "(Modelo bilineal – Unidades: Tonf y m)",

    # ✅ Box
    "b4_box_hdr": "=== PROPIEDADES DEL AISLADOR LRB INDIVIDUAL ===",
    "b4_box_keff": "Rigidez efectiva 𝐊ₑ𝑓𝑓",
    "b4_box_ceq":  "Amortiguamiento viscoso 𝐂ₑ𝑞",
    "b4_box_ke":   "Rigidez inicial elástica 𝐊ₑ",
    "b4_box_kp":   "Rigidez postfluencia 𝐊ₚ",
    "b4_box_fy":   "Fuerza de fluencia 𝐅ᵧ",
    "b4_box_r":    "Relación de rigideces 𝐫 = 𝐊ₚ/𝐊ₑ",

    # ✅ Hysteresis texts
    "b4_hyst_title": "Ciclo histerético bilineal – Aislador LRB",
    "b4_hyst_xlabel": "Desplazamiento Δ (m)",
    "b4_hyst_ylabel": "Fuerza cortante F (Tonf)",

    # ✅ Modal table headers
    "b4_mode": "Modo",
    "b4_f": "f [Hz]",
    "b4_T": "T [s]",
    
    # ✅ Warning text prefix
    "b4_warn_hdr": "Advertencia de período objetivo",

    "b4_box_Tm": "Período efectivo 𝐓ₘ",
    "b4_box_betaM": "Amortiguamiento efectivo βₘ",
    "b4_box_DM": "Desplazamiento máximo 𝐃ₘ",
    "b4_box_dy": "Desplazamiento de fluencia δᵧ",
    "b4_box_DL": "Diámetro del núcleo de plomo 𝐃ₗ",
    "b4_box_DB": "Diámetro del aislador 𝐃ᵦ",
    "b4_box_tr": "Espesor total de caucho 𝐭ᵣ",

    "b4_box_chk_dy": "δᵧ < Dₘ",
    "b4_box_chk_k": "Kₑ > Kₚ",
    "b4_box_yes": "Sí",
    "b4_box_no": "No",
})

# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
st.markdown(f"## 🧱 {tr('b4_title')}")

# -------------------------------------------------------------------------
# ✅ Validación previa
# -------------------------------------------------------------------------
if ("K_cond" not in st.session_state) or ("M_cond" not in st.session_state):
    st.info(tr("b4_need_prev"))
    st.stop()

Kc = st.session_state["K_cond"]
Mc = st.session_state["M_cond"]

# -------------------------------------------------------------------------
# ✅ Modal cacheado: solo recalcula si cambia K/M
# -------------------------------------------------------------------------
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

    w = np.sqrt(np.maximum(w2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        T_modos = np.where(w > 0, 2*np.pi / w, np.inf)
        f_modos = np.where(T_modos != np.inf, 1 / T_modos, 0.0)

    st.session_state["_km_key_modal"] = km_key
    st.session_state["w_sin"] = w
    st.session_state["T_sin"] = T_modos
    st.session_state["f_sin"] = f_modos
    st.session_state["phi_norm_sin"] = phi_norm
    st.session_state["v_norm_sin"]   = phi_norm

# Recuperar modal
w        = st.session_state["w_sin"]
T_modos  = st.session_state["T_sin"]
f_modos  = st.session_state["f_sin"]

# -------------------------------------------------------------------------
# ✅ Layout 2 columnas del bloque
# -------------------------------------------------------------------------
col_izq, col_der = st.columns([1.3, 1.7], gap="large")

with col_izq:
    st.markdown(f"### ⚙️ {tr('b4_subtitle')}")
    st.caption(tr("b4_zeta_note"))

    ζ = 0.05
    st.session_state["zeta_modal"] = float(ζ)

    # =========================================================
    # ✅ MENSAJES PERSISTENTES (para que no se pierdan con rerun)
    # =========================================================
    if st.session_state.get("b4_last_status") == "ok":
        st.success(tr("b4_ok"))
    elif st.session_state.get("b4_last_status") == "err":
        st.error(st.session_state.get("b4_last_err_msg", "—"))

    if st.session_state.get("b4_warning_msg"):
        st.warning(st.session_state["b4_warning_msg"])

    # =========================
    # Rayleigh cacheado
    # =========================
    ray_key = (km_key, float(ζ))
    if st.session_state.get("_ray_key") != ray_key:
        w_use = np.asarray(w, dtype=float).ravel()

        if (
            len(w_use) >= 2
            and np.isfinite(w_use[0]) and np.isfinite(w_use[1])
            and w_use[0] > 0 and w_use[1] > 0
        ):
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

    # =========================
    # Modal
    # =========================
    with st.expander(tr("b4_modal_data"), expanded=False):
        st.caption(tr("h_b4_modal_data"))
        tabla = np.vstack((np.arange(1, len(f_modos) + 1), f_modos, T_modos)).T
        df_modos = pd.DataFrame(
            tabla,
            columns=[tr("b4_mode"), tr("b4_f"), tr("b4_T")]
        ).round(5)

        st.dataframe(
            df_modos,
            hide_index=True,
            use_container_width=True,
            height=min(35 * (len(df_modos) + 1), 160),
        )

    st.markdown("---")
    st.markdown(f"#### 🧰 {tr('b4_design_mode_hdr')}")

    # =========================
    # Layout controles + checks
    # =========================
    c_ctrl = st.container()

    with c_ctrl:
        modo = st.radio(
            tr("b4_design_mode"),
            [tr("b4_mode_auto"), tr("b4_mode_Tobj")],
            index=0,
            key="modo_lrb",
            help=tr("h_b4_design_mode"),
        )
        modo_automatico = (modo == tr("b4_mode_auto"))
        modo_periodo_objetivo = (modo == tr("b4_mode_Tobj"))
    
        T_objetivo = st.number_input(
            tr("b4_Tobj"),
            0.5, 5.0, 2.5, 0.1,
            key="T_obj_lrb",
            disabled=modo_automatico,
            help=tr("h_b4_Tobj"),
        )
    
        ejecutar = st.button(
            f"⚙️ {tr('b4_run')}",
            key="btn_lrb",
            help=tr("h_b4_run"),
        )

    # =========================
    # Botón cálculo (con rerun para actualizar checks inmediato)
    # =========================
    if ejecutar:
        # limpiar estados persistentes
        st.session_state["b4_last_status"] = None
        st.session_state["b4_last_err_msg"] = None
        st.session_state["b4_warning_msg"] = None

        faltantes = []
        for k in ["SD1", "SDS", "T_sin", "nodes"]:
            val = st.session_state.get(k)
            if val is None or (isinstance(val, (list, np.ndarray)) and len(val) == 0):
                faltantes.append(k)

        if faltantes:
            st.session_state["b4_last_status"] = "err"
            st.session_state["b4_last_err_msg"] = tr("b4_missing_prev").format(keys=", ".join(faltantes))
            st.rerun()

        try:
            resultados_ais = diseno_aislador_LRB(
                SD1=st.session_state["SD1"],
                SDS=st.session_state["SDS"],
                T_sin=st.session_state["T_sin"],
                Mc=Mc,
                nodos_restringidos=[
                    nid
                    for (x, y, nid) in st.session_state["nodes"]
                    if float(y) == 0.0
                ],
                modo_automatico=modo_automatico,
                modo_periodo_objetivo=modo_periodo_objetivo,
                T_objetivo=T_objetivo,
            )

            req = [
                "k_inicial_1ais",
                "k_post_1ais",
                "yield_1ais",
                "c_1ais",
                "keff_1ais",
                "D_M",
                "beta_M",
                "delta_y",  # ✅ dy real
            ]

            faltan = [k for k in req if k not in resultados_ais]
            if faltan:
                st.session_state["b4_last_status"] = "err"
                st.session_state["b4_last_err_msg"] = tr("b4_missing_res").format(keys=faltan)
                st.rerun()

            st.session_state["res_aislador"] = resultados_ais
            for k in req:
                st.session_state[k] = float(resultados_ais[k])

            # ✅ Guardar warning persistente (si viene desde funciones_usuario.py)
            if bool(resultados_ais.get("warning_periodo_bajo", False)):
                msg = resultados_ais.get("mensaje_warning", None)
                if msg:
                    st.session_state["b4_warning_msg"] = f"⚠️ {tr('b4_warn_hdr')}: {msg}"

            st.session_state["b4_last_status"] = "ok"
            st.rerun()

        except Exception as e:
            st.session_state["b4_last_status"] = "err"
            st.session_state["b4_last_err_msg"] = tr("b4_err").format(e=e)
            st.rerun()

    # -------------------------------------------------------------------------
    # ✅ N aisladores (mismo criterio que usas en el diseño: nodos con y == 0.0)
    # -------------------------------------------------------------------------
    nodes_now = st.session_state.get("nodes", [])
    nodos_restringidos_now = [
        nid for (x, y, nid) in nodes_now
        if float(y) == 0.0
    ]
    n_aisladores_now = int(len(nodos_restringidos_now))
    st.session_state["n_aisladores"] = n_aisladores_now
    st.session_state["nodos_restringidos"] = nodos_restringidos_now

    # =========================
    # Resumen fijo (persistente) + N y N*Keff
    # =========================
    if "res_aislador" in st.session_state:
        r = st.session_state["res_aislador"]
    
        # ✅ Valores principales
        T_M = float(r.get("T_M", np.nan))
        beta_M = float(r.get("beta_M", np.nan))
        beta_M_pct = 100.0 * beta_M if np.isfinite(beta_M) else np.nan
    
        keff_1ais = float(r.get("keff_1ais", np.nan))
        c_1ais = float(r.get("c_1ais", np.nan))
        k_ini_1ais = float(r.get("k_inicial_1ais", np.nan))
        k_post_1ais = float(r.get("k_post_1ais", np.nan))
        fy_1ais = float(r.get("yield_1ais", np.nan))
    
        ratio_post = (
            float(k_post_1ais / k_ini_1ais)
            if (np.isfinite(k_ini_1ais) and k_ini_1ais > 0)
            else np.nan
        )
    
        D_M = float(r.get("D_M", np.nan))
        delta_y = float(r.get("delta_y", np.nan))
        D_L = float(r.get("D_L", np.nan))
        D_B = float(r.get("D_B", np.nan))
        t_r = float(r.get("t_r", np.nan))
    
        # ✅ Verificaciones mínimas útiles
        ok_dy = np.isfinite(delta_y) and np.isfinite(D_M) and (delta_y > 0) and (delta_y < D_M)
        ok_k = np.isfinite(k_ini_1ais) and np.isfinite(k_post_1ais) and (k_ini_1ais > k_post_1ais > 0)
    
        yes_txt = tr("b4_box_yes")
        no_txt = tr("b4_box_no")
    
        # ✅ guardar para usar en Bloque 5/6
        st.session_state["keff_1ais"] = float(keff_1ais) if np.isfinite(keff_1ais) else np.nan
    
        st.markdown(f"""
        <div style="
            background-color:#1E2331;
            color:#F4F6FA;
            padding:10px 12px;
            border-radius:10px;
            border:1px solid #3A4050;
            font-family:Consolas, monospace;
            margin-top:10px;">
    
            <div style="font-weight:700; margin-bottom:10px;">
                {tr("b4_box_hdr")}
            </div>
    
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:flex-start;
                gap:24px;
                margin-bottom:8px;">
    
                <!-- COLUMNA IZQUIERDA -->
                <div style="
                    flex:1.2;
                    min-width:260px;
                    line-height:1.40;">
                    {tr("b4_box_Tm")} : {T_M:.3f} s<br>
                    {tr("b4_box_betaM")} : {beta_M_pct:.2f} %<br>
                    {tr("b4_box_DM")} : {D_M:.4f} m<br>
                    {tr("b4_box_dy")} : {delta_y:.4f} m<br>
                    <br>
    
                    {tr("b4_box_keff")} : {keff_1ais:.3f} Tonf/m<br>
                    {tr("b4_box_ceq")} : {c_1ais:.3f} Tonf·s/m<br>
                    <br>
    
                    {tr("b4_box_chk_dy")} : {yes_txt if ok_dy else no_txt}<br>
                    {tr("b4_box_chk_k")} : {yes_txt if ok_k else no_txt}
                </div>
    
                <!-- COLUMNA DERECHA -->
                <div style="
                    flex:1.0;
                    min-width:240px;
                    line-height:1.40;
                    padding-left:6px;">
                    {tr("b4_box_ke")} : {k_ini_1ais:.3f} Tonf/m<br>
                    {tr("b4_box_kp")} : {k_post_1ais:.3f} Tonf/m<br>
                    {tr("b4_box_fy")} : {fy_1ais:.3f} Tonf<br>
                    {tr("b4_box_r")} : {ratio_post:.3f}<br>
                    <br>
    
                    {tr("b4_box_DL")} : {D_L:.4f} m<br>
                    {tr("b4_box_DB")} : {D_B:.4f} m<br>
                    {tr("b4_box_tr")} : {t_r:.4f} m
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_der:
    if "res_aislador" not in st.session_state:
        st.markdown(f"### 📊 {tr('b4_plot_hdr')}")
        st.markdown(f"**{tr('b4_plot_sub')}**")
        st.info(tr("b4_need_run_plot"))
    else:
        resultados_ais = st.session_state["res_aislador"]

        st.markdown(f"### 📊 {tr('b4_plot_hdr')}")
        st.markdown(f"**{tr('b4_plot_sub')}**")

        # Tu estilo + plot
        set_style_arctic_dark()

        fig, ax = plot_ciclo_histeretico_lrb(
            Ke=resultados_ais["k_inicial_1ais"],
            Kp=resultados_ais["k_post_1ais"],
            Fy=resultados_ais["yield_1ais"],
            dy=resultados_ais["delta_y"],   # ✅ dy REAL (yield displacement)
            D2=resultados_ais["D_M"],
            Keff_ref=resultados_ais["keff_1ais"],
            titulo=tr("b4_hyst_title"),
            xlabel=tr("b4_hyst_xlabel"),
            ylabel=tr("b4_hyst_ylabel"),
        )
        st.pyplot(fig, use_container_width=True)

# =============================================================================
# ========= BLOQUE 5: MODAL + ESQUEMAS OPTIMIZADO PARA HASTA 30 PISOS =========
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from numpy.linalg import solve

from funciones_usuario import (
    modal_props,
    plot_modes_grid,
    _force_same_height_modes,
    plot_modelo_condensado_fijo,
    plot_modelo_condensado_aislado,
)

# -------------------------------------------------------------------------
# ✅ Textos EN/ES (solo para este bloque) + HELPERS (SIN cambiar lógica)
# -------------------------------------------------------------------------
T["en"].update({
    "b5_title": "Modal analysis – Fixed base vs Base isolated",

    "b5_need_fix": "First generate the FIXED model (K_cond, M_cond).",
    "b5_need_geom": "First generate the structural model (nodes, element_node_pairs, propiedades).",
    "b5_need_iso": "First design the isolator (res_aislador).",

    "b5_mats_hdr": "Condensed matrices",
    "b5_fix_mats": "Fixed-base model matrices",
    "b5_iso_mats": "Isolated-base model matrices",

    "b5_fix_Kg": "Fixed global stiffness",
    "b5_fix_K": "Fixed condensed stiffness",
    "b5_fix_M": "Fixed condensed mass",

    "b5_iso_Kg": "Isolated global stiffness",
    "b5_iso_K": "Isolated condensed stiffness",
    "b5_iso_M": "Isolated condensed mass",

    "b5_no_fix_Kg": "Fixed global stiffness matrix was not found in session_state.",
    "b5_no_iso_Kg": "Isolated global stiffness matrix was not generated.",

    "b5_modal_hdr": "Modal analysis",
    "b5_modes_fix": "Mode shapes – FIXED structure",
    "b5_modes_iso": "Mode shapes – ISOLATED structure",

    "b5_col_fix_1": "1️⃣ Frequencies and periods – FIXED",
    "b5_col_fix_2": "2️⃣ Mode shapes – FIXED",
    "b5_col_fix_3": "3️⃣ Model scheme – FIXED",

    "b5_col_iso_1": "1️⃣ Frequencies and periods – ISOLATED",
    "b5_col_iso_2": "2️⃣ Mode shapes – ISOLATED",
    "b5_col_iso_3": "3️⃣ Model scheme – ISOLATED",

    "b5_mode": "Mode",
    "b5_f": "f [Hz]",
    "b5_T": "T [s]",

    "b5_err_ais": "Error while generating ISOLATED matrices: {e}",
    "b5_done": "Modal analysis ready.",

    "h_b5_mats": "Here you can see the global stiffness matrix and the condensed matrices (1 lateral DOF per floor) used for modal analysis.",
    "h_b5_modal": "Mode shapes are normalized for plotting; periods come from the eigenvalue solution.",

    "b5_no_modes": "No modes to plot",
    "b5_mode_lbl": "Mode",
    "b5_height_lbl": "Height [m]",
    "b5_fixed_base_lbl": "Fixed base",
    "b5_fixed_model_lbl": "Condensed FIXED model",
    "b5_iso_model_lbl": "Condensed ISOLATED model",

    "b5_mpart": "Mpart [%]",
    "b5_macc": "Macc [%]",
})

T["es"].update({
    "b5_title": "Análisis modal – Base fija vs Base aislada",

    "b5_need_fix": "⚙️ Primero genera el modelo FIJO (K_cond, M_cond).",
    "b5_need_geom": "⚙️ Primero genera el modelo estructural (nodes, element_node_pairs, propiedades).",
    "b5_need_iso": "⚙️ Primero diseña el aislador (res_aislador).",

    "b5_mats_hdr": "📘 Matrices condensadas",
    "b5_fix_mats": "🧱 Matrices del modelo FIJO",
    "b5_iso_mats": "🟩 Matrices del modelo AISLADO",

    "b5_fix_Kg": "Rigidez global FIJA",
    "b5_fix_K": "Rigidez condensada FIJA",
    "b5_fix_M": "Masa condensada FIJA",

    "b5_iso_Kg": "Rigidez global AISLADA",
    "b5_iso_K": "Rigidez condensada AISLADA",
    "b5_iso_M": "Masa condensada AISLADA",

    "b5_no_fix_Kg": "No se encontró la matriz de rigidez global FIJA en session_state.",
    "b5_no_iso_Kg": "No se generó la matriz de rigidez global AISLADA.",

    "b5_modal_hdr": "📊 Análisis modal",
    "b5_modes_fix": "Modos de Vibración – Estructura FIJA",
    "b5_modes_iso": "Modos de Vibración – Estructura AISLADA",

    "b5_col_fix_1": "1️⃣ Frecuencias y períodos – FIJA",
    "b5_col_fix_2": "2️⃣ Modos de vibración – FIJA",
    "b5_col_fix_3": "3️⃣ Esquema del modelo – FIJA",

    "b5_col_iso_1": "1️⃣ Frecuencias y períodos – AISLADA",
    "b5_col_iso_2": "2️⃣ Modos de vibración – AISLADA",
    "b5_col_iso_3": "3️⃣ Esquema del modelo – AISLADA",

    "b5_mode": "Modo",
    "b5_f": "f [Hz]",
    "b5_T": "T [s]",

    "b5_err_ais": "❌ Error al generar matrices AISLADAS: {e}",
    "b5_done": "✅ Análisis modal listo.",

    "h_b5_mats": "Aquí puedes ver la matriz de rigidez global y las matrices condensadas (1 GDL lateral por piso) usadas para el análisis modal.",
    "h_b5_modal": "Las formas modales se normalizan para los gráficos; los períodos se obtienen de la solución por autovalores.",

    "b5_no_modes": "Sin modos para graficar",
    "b5_mode_lbl": "Modo",
    "b5_height_lbl": "Altura [m]",
    "b5_fixed_base_lbl": "Base fija",
    "b5_fixed_model_lbl": "Modelo Condensado FIJO",
    "b5_iso_model_lbl": "Modelo Condensado AISLADO",

    "b5_mpart": "Mpart [%]",
    "b5_macc": "Macc [%]",
})

# ✅ CLAVE: compartir T para que funciones_usuario.tr() funcione
st.session_state["T"] = T

# -------------------------------------------------------------------------
# ✅ Helper: masa modal participante por modo [%]
# -------------------------------------------------------------------------
def modal_participating_mass_percent(M, Vn):
    """
    Calcula el porcentaje de masa modal participante por modo, usando:
        Gamma_r = (phi_r^T M 1) / (phi_r^T M phi_r)
        M*_r    = (phi_r^T M 1)^2 / (phi_r^T M phi_r)
        %M_r    = 100 * M*_r / (1^T M 1)

    Parámetros
    ----------
    M : array_like, shape (n, n)
        Matriz de masas.
    Vn : array_like, shape (n, m)
        Matriz modal; cada columna es un modo.

    Retorna
    -------
    mpart_pct : ndarray, shape (m,)
        Porcentaje de masa participante por modo.
    """
    M = np.asarray(M, dtype=float)
    Vn = np.asarray(Vn, dtype=float)

    n = M.shape[0]
    r = np.ones(n, dtype=float)

    m_total = float(r.T @ M @ r)
    if not np.isfinite(m_total) or abs(m_total) < 1e-14:
        return np.zeros(Vn.shape[1], dtype=float)

    mpart_pct = []
    for i in range(Vn.shape[1]):
        phi = Vn[:, i].reshape(-1, 1)

        num = (phi.T @ M @ r.reshape(-1, 1)).item()
        den = (phi.T @ M @ phi).item()

        if (not np.isfinite(den)) or abs(den) < 1e-14:
            m_eff = 0.0
        else:
            m_eff = (num ** 2) / den

        pct = 100.0 * m_eff / m_total
        mpart_pct.append(pct)

    return np.asarray(mpart_pct, dtype=float)

# -------------------------------------------------------------------------
# ✅ Helper: masa modal acumulada [%]
# -------------------------------------------------------------------------
def modal_cumulative_mass_percent(mpart_pct):
    """
    Retorna la masa modal acumulada en porcentaje.
    """
    mpart_pct = np.asarray(mpart_pct, dtype=float)
    return np.cumsum(mpart_pct)
    
# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
st.markdown(f"## 🌊 {tr('b5_title')}")

# -----------------------------------------------------------------
# ✅ PRERREQUISITOS
# -----------------------------------------------------------------
if "K_cond" not in st.session_state or "M_cond" not in st.session_state:
    st.info(tr("b5_need_fix"))
    st.stop()

if "nodes" not in st.session_state or "element_node_pairs" not in st.session_state or "propiedades" not in st.session_state:
    st.info(tr("b5_need_geom"))
    st.stop()

if "res_aislador" not in st.session_state:
    st.info(tr("b5_need_iso"))
    st.stop()

K_fix = st.session_state["K_cond"]
M_fix = st.session_state["M_cond"]

# ✅ intenta leer K global fija si existe
K_global_fix = st.session_state.get("K_global", None)

nodes              = st.session_state["nodes"]
element_node_pairs = st.session_state["element_node_pairs"]
propiedades        = st.session_state["propiedades"]

res_ais   = st.session_state["res_aislador"]
keff_1ais = float(res_ais["keff_1ais"])

# ✅ N aisladores: leer desde session_state (viene del Bloque 4)
n_aisladores = st.session_state.get("n_aisladores", None)
if n_aisladores is None:
    n_aisladores = sum(1 for (x, y, nid) in nodes if abs(float(y)) < 1e-9)
n_aisladores = int(n_aisladores)
st.session_state["n_aisladores"] = n_aisladores

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
    gdl_ss = sorted({
        dof for (_nid, gtipo), dof in gdl_map_libre.items()
        if (gtipo in ("vy", "theta")) and (dof is not None)
    })

    kpp = K_global_libre[np.ix_(gdl_pp, gdl_pp)]
    kss = K_global_libre[np.ix_(gdl_ss, gdl_ss)]
    kps = K_global_libre[np.ix_(gdl_pp, gdl_ss)]
    ksp = K_global_libre[np.ix_(gdl_ss, gdl_pp)]

    K_vx_nodo = kpp - kps @ solve(kss, ksp)

    nodes_arr = np.array(nodes, dtype=float)
    pisos_y = np.sort(np.unique(nodes_arr[:, 1][nodes_arr[:, 1] > 0]))
    niveles_y = np.insert(pisos_y, 0, 0.0)
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

    # ✅ AISLADOR: resorte a tierra en el DOF 0
    k_iso_total = keff_1ais * n_aisladores
    K_cond_ais[0, 0] += k_iso_total

    # ✅ tomar b_col_x real desde Sección 1
    param_estruct_b5 = st.session_state.get("param_estruct", {})
    modo_avanzado_b5 = bool(param_estruct_b5.get("modo_avanzado", False))
    b_col_cm_b5 = param_estruct_b5.get("b_col_cm", None)

    if (not modo_avanzado_b5) and (b_col_cm_b5 is not None):
        b_col_x_b5 = float(b_col_cm_b5) / 100.0
    else:
        b_col_x_b5 = 0.0

    M_cond_ais = calcular_matriz_masas_con_aislador(
        nodes,
        element_node_pairs,
        propiedades,
        peso_especifico=st.session_state["peso_especifico"],
        sobrecarga_muerta=st.session_state["sobrecarga_muerta"],
        b_col_x=b_col_x_b5,
    )

    st.session_state["K_global_libre"] = np.array(K_global_libre, copy=True)
    st.session_state["K_cond_ais"] = np.array(K_cond_ais, copy=True)
    st.session_state["M_cond_ais"] = np.array(M_cond_ais, copy=True)

except Exception as e:
    st.error(tr("b5_err_ais").format(e=e))
    st.stop()

# -----------------------------------------------------------------
# 5B) MATRICES (EXPANDERS)
# -----------------------------------------------------------------
st.subheader(tr("b5_mats_hdr"))
st.caption(tr("h_b5_mats"))

colM1, colM2 = st.columns([1, 1], gap="large")

with colM1:
    with st.expander(tr("b5_fix_mats"), expanded=False):

        if K_global_fix is not None:
            K_global_fix_arr = np.asarray(K_global_fix, float)
            st.markdown(
                f"**{tr('b5_fix_Kg')}** "
                f"(dim = {K_global_fix_arr.shape[0]}×{K_global_fix_arr.shape[1]}):"
            )
            st.dataframe(
                pd.DataFrame(np.round(K_global_fix_arr, 3)),
                use_container_width=True
            )
        else:
            st.info(tr("b5_no_fix_Kg"))

        st.markdown(
            f"**{tr('b5_fix_K')}** "
            f"(dim = {np.asarray(K_fix).shape[0]}×{np.asarray(K_fix).shape[1]}):"
        )
        st.dataframe(
            pd.DataFrame(np.round(np.asarray(K_fix, float), 3)),
            use_container_width=True
        )

        st.markdown(
            f"**{tr('b5_fix_M')}** "
            f"(dim = {np.asarray(M_fix).shape[0]}×{np.asarray(M_fix).shape[1]}):"
        )
        st.dataframe(
            pd.DataFrame(np.round(np.asarray(M_fix, float), 5)),
            use_container_width=True
        )

with colM2:
    with st.expander(tr("b5_iso_mats"), expanded=False):

        if "K_global_libre" in st.session_state:
            K_global_ais_arr = np.asarray(st.session_state["K_global_libre"], float)
            st.markdown(
                f"**{tr('b5_iso_Kg')}** "
                f"(dim = {K_global_ais_arr.shape[0]}×{K_global_ais_arr.shape[1]}):"
            )
            st.dataframe(
                pd.DataFrame(np.round(K_global_ais_arr, 3)),
                use_container_width=True
            )
        else:
            st.info(tr("b5_no_iso_Kg"))

        st.markdown(
            f"**{tr('b5_iso_K')}** "
            f"(dim = {K_cond_ais.shape[0]}×{K_cond_ais.shape[1]}):"
        )
        st.dataframe(
            pd.DataFrame(np.round(K_cond_ais, 3)),
            use_container_width=True
        )

        st.markdown(
            f"**{tr('b5_iso_M')}** "
            f"(dim = {M_cond_ais.shape[0]}×{M_cond_ais.shape[1]}):"
        )
        st.dataframe(
            pd.DataFrame(np.round(M_cond_ais, 5)),
            use_container_width=True
        )

# -----------------------------------------------------------------
# 5C) ANÁLISIS MODAL (SIMÉTRICO)
# -----------------------------------------------------------------
st.subheader(tr("b5_modal_hdr"))
st.caption(tr("h_b5_modal"))

# ✅ Ambos gráficos arrancan en y = 0.0
niveles_fix = np.insert(pisos_y, 0, 0.0)
niveles_ais = np.insert(pisos_y, 0, 0.0)

# ✅ Resolver modal primero
w_fix, T_fix, f_fix, Vn_fix = modal_props(np.asarray(K_fix, float), np.asarray(M_fix, float))
w_ais, T_ais, f_ais, Vn_ais = modal_props(K_cond_ais, M_cond_ais)

mpart_fix = modal_participating_mass_percent(np.asarray(M_fix, float), Vn_fix)
mpart_ais = modal_participating_mass_percent(np.asarray(M_cond_ais, float), Vn_ais)

# ✅ acumulada
macc_fix = modal_cumulative_mass_percent(mpart_fix)
macc_ais = modal_cumulative_mass_percent(mpart_ais)

# ✅ Guardar en session_state
st.session_state["w_sin"] = w_fix
st.session_state["T_sin"] = T_fix
st.session_state["v_norm_sin"] = Vn_fix

st.session_state["w_ais"] = w_ais
st.session_state["T_ais"] = T_ais
st.session_state["v_norm_ais"] = Vn_ais

st.session_state["mpart_fix"] = mpart_fix
st.session_state["macc_fix"] = macc_fix

st.session_state["mpart_ais"] = mpart_ais
st.session_state["macc_ais"] = macc_ais

n_modos_fix = int(Vn_fix.shape[1])
n_modos_ais = int(Vn_ais.shape[1])

# ✅ Graficar después de tener Vn_fix y Vn_ais
fig_fix_modes = plot_modes_grid(
    Vn_fix,
    niveles_fix,
    T_fix,
    tr("b5_modes_fix"),
    include_base_minus1=False,
    ncols=6
)

fig_ais_modes = plot_modes_grid(
    Vn_ais,
    niveles_ais,
    T_ais,
    tr("b5_modes_iso"),
    include_base_minus1=False,
    ncols=6
)

fig_fix_modes, fig_ais_modes = _force_same_height_modes(
    fig_fix_modes,
    fig_ais_modes,
    nA=n_modos_fix,
    nB=n_modos_ais,
    ncols=6
)

fig_fix_scheme = plot_modelo_condensado_fijo(
    np.asarray(K_fix, float),
    np.asarray(M_fix, float),
    niveles_fix,
    pisos_y
)

fig_ais_scheme = plot_modelo_condensado_aislado(
    K_cond_ais,
    M_cond_ais,
    pisos_y
)

colL, colR = st.columns([1, 1], gap="large")

with colL:
    with st.container(border=True):
        st.subheader(tr("b5_col_fix_1"))
        tabla_fix = pd.DataFrame({
            tr("b5_mode"): np.arange(1, len(f_fix) + 1),
            tr("b5_f"): np.round(f_fix, 5),
            tr("b5_T"): np.round(T_fix, 5),
            tr("b5_mpart"): np.round(mpart_fix, 3),
            tr("b5_macc"): np.round(macc_fix, 3),
        })
        st.dataframe(tabla_fix, hide_index=True, use_container_width=True, height=170)

    with st.container(border=True):
        st.subheader(tr("b5_col_fix_2"))
        st.pyplot(fig_fix_modes, use_container_width=True)
        plt.close(fig_fix_modes)

    with st.container(border=True):
        st.subheader(tr("b5_col_fix_3"))
        st.pyplot(fig_fix_scheme, use_container_width=True)
        plt.close(fig_fix_scheme)

with colR:
    with st.container(border=True):
        st.subheader(tr("b5_col_iso_1"))
        tabla_ais = pd.DataFrame({
            tr("b5_mode"): np.arange(1, len(f_ais) + 1),
            tr("b5_f"): np.round(f_ais, 5),
            tr("b5_T"): np.round(T_ais, 5),
            tr("b5_mpart"): np.round(mpart_ais, 3),
            tr("b5_macc"): np.round(macc_ais, 3),
        })
        st.dataframe(tabla_ais, hide_index=True, use_container_width=True, height=170)

    with st.container(border=True):
        st.subheader(tr("b5_col_iso_2"))
        st.pyplot(fig_ais_modes, use_container_width=True)
        plt.close(fig_ais_modes)

    with st.container(border=True):
        st.subheader(tr("b5_col_iso_3"))
        st.pyplot(fig_ais_scheme, use_container_width=True)
        plt.close(fig_ais_scheme)

st.success(tr("b5_done"))

# =============================================================================
# === BLOQUE 6: ANÁLISIS DINÁMICO (NEWMARK-β) SIMÉTRICO =======================
# =============================================================================
import numpy as np
import streamlit as st
import pandas as pd

from funciones_usuario import (
    rayleigh_from_w,
    pick_two_w,
    ensure_2d,
    modal_w,
    _sig,
    make_excel_per_floor,
)

# -------------------------------------------------------------------------
# ✅ DEFENSIVO: si no existen T / tr, no revienta (solo en este bloque)
# -------------------------------------------------------------------------
if "T" not in globals():
    T = {"es": {}, "en": {}}

if "tr" not in globals():
    def tr(k: str) -> str:
        return k

# -------------------------------------------------------------------------
# ✅ Textos EN/ES (solo para este bloque)
# -------------------------------------------------------------------------
T["en"].update({
    "b6_title": "Dynamic analysis (Newmark-β) – FIXED linear vs ISOLATED linear",
    "b6_need_prev": "⚙️ Missing in session_state: {keys}. Run previous blocks.",
    "b6_need_newmark": "❌ Function `newmark` was not found (neither in globals nor in session_state).",
    "b6_need_plotfun": "❌ Function `graficar_respuesta_por_piso` was not found (neither in globals nor in session_state).",
    "b6_zeta_note": "Using modal damping ζ = **{zeta:.3f}**.",
    "b6_left_hdr": "🧱 FIXED",
    "b6_right_hdr": "🟩 ISOLATED",
    "b6_metrics_dur": "Duration",
    "b6_left_resp": "📈 FIXED – Level responses",
    "b6_right_resp": "📈 ISOLATED – Level responses",
    "b6_floor": "Level {i}",
    "b6_iso_level": "Isolator (Level 0)",
    "b6_ok": "✅ Dynamic analysis ready",
    "h_b6_title": "Both cases use linear Newmark-β with Rayleigh damping. The isolated case employs the condensed effective-stiffness model assembled in previous blocks.",
    "b6_g_u": "Displacement",
    "b6_g_v": "Velocity",
    "b6_g_a": "Acceleration",
    "b6_g_time": "Time [s]",
    "b6_g_u_y": "u [m]",
    "b6_g_v_y": "v [m/s]",
    "b6_g_a_y": "a [m/s²]",
    "b6_dl_xlsx": "Download Excel data",
    "b6_dl_help": "Downloads an .xlsx file with one sheet per level: t, a, v, u.",
})

T["es"].update({
    "b6_title": "Análisis dinámico (Newmark-β) – FIJA lineal vs AISLADA lineal",
    "b6_need_prev": "⚙️ Faltan en session_state: {keys}. Ejecuta bloques anteriores.",
    "b6_need_newmark": "❌ No se encontró la función `newmark` (ni en globals ni en session_state).",
    "b6_need_plotfun": "❌ No se encontró `graficar_respuesta_por_piso` (ni en globals ni en session_state).",
    "b6_zeta_note": "Usando amortiguamiento modal ζ = **{zeta:.3f}**.",
    "b6_left_hdr": "🧱 FIJA",
    "b6_right_hdr": "🟩 AISLADA",
    "b6_metrics_dur": "Duración",
    "b6_left_resp": "📈 FIJA – Respuestas por nivel",
    "b6_right_resp": "📈 AISLADA – Respuestas por nivel",
    "b6_floor": "Nivel {i}",
    "b6_iso_level": "Aislador (Nivel 0)",
    "b6_ok": "✅ Análisis dinámico listo",
    "h_b6_title": "Ambos casos usan Newmark-β lineal con amortiguamiento Rayleigh. El caso aislado emplea el modelo condensado con rigidez efectiva ya ensamblado en bloques previos.",
    "b6_g_u": "Desplazamiento",
    "b6_g_v": "Velocidad",
    "b6_g_a": "Aceleración",
    "b6_g_time": "Tiempo [s]",
    "b6_g_u_y": "u [m]",
    "b6_g_v_y": "v [m/s]",
    "b6_g_a_y": "a [m/s²]",
    "b6_dl_xlsx": "Descargar datos Excel",
    "b6_dl_help": "Descarga un archivo .xlsx con una pestaña por nivel: t, a, v, u.",
})

# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
st.subheader(f"⚙️ {tr('b6_title')}")
st.caption(tr("h_b6_title"))

# -----------------------------------------------------------------
# ✅ PRERREQUISITOS (mínimos y coherentes con este bloque)
# -----------------------------------------------------------------
req_keys = ["K_cond", "M_cond", "K_cond_ais", "M_cond_ais", "ag_filt", "t_ag", "dt"]
missing = [k for k in req_keys if k not in st.session_state]
if missing:
    st.info(tr("b6_need_prev").format(keys=", ".join(missing)))
    st.stop()

def get_fun(name):
    f = st.session_state.get(name, None)
    if f is None and name in globals():
        f = globals()[name]
    return f

newmark = get_fun("newmark")
graficar_respuesta_por_piso = get_fun("graficar_respuesta_por_piso")

if newmark is None:
    st.error(tr("b6_need_newmark"))
    st.stop()
if graficar_respuesta_por_piso is None:
    st.error(tr("b6_need_plotfun"))
    st.stop()

# -----------------------------------------------------------------
# ✅ Wrapper: intenta pasar labels traducidos (sin romper si tu función no acepta)
# -----------------------------------------------------------------
def _plot_resp(t, u, v, a, alturas, t_total, nombre_piso):
    kwargs = dict(
        titulo_u=tr("b6_g_u"),
        titulo_v=tr("b6_g_v"),
        titulo_a=tr("b6_g_a"),
        xlabel=tr("b6_g_time"),
        ylabel_u=tr("b6_g_u_y"),
        ylabel_v=tr("b6_g_v_y"),
        ylabel_a=tr("b6_g_a_y"),
    )
    try:
        return graficar_respuesta_por_piso(
            t, u, v, a, alturas, t_total,
            nombre_piso=nombre_piso,
            **kwargs
        )
    except TypeError:
        return graficar_respuesta_por_piso(
            t, u, v, a, alturas, t_total,
            nombre_piso=nombre_piso
        )

# -----------------------------------------------------------------
# ✅ Lectura de datos
# -----------------------------------------------------------------
K_fix = np.array(st.session_state["K_cond"], dtype=float)
M_fix = np.array(st.session_state["M_cond"], dtype=float)

# ✅ AISLADA: modelo condensado con rigidez efectiva ya ensamblado (NO tocar)
K_ais_eff = np.array(st.session_state["K_cond_ais"], dtype=float)
M_ais_eff = np.array(st.session_state["M_cond_ais"], dtype=float)

ag_filt = np.asarray(st.session_state["ag_filt"], float).ravel()
t_ag    = np.asarray(st.session_state["t_ag"], float).ravel()
dt      = float(st.session_state["dt"])

# -----------------------------------------------------------------
# ✅ ζ desde Bloque 4 (o default 5%) — SIN pedir input
# -----------------------------------------------------------------
zeta = float(st.session_state.get("zeta_modal", 0.05))
st.caption(tr("b6_zeta_note").format(zeta=zeta))
st.session_state["zeta_modal"] = float(zeta)

# -----------------------------------------------------------------
# ✅ Newmark (promedio constante)
# -----------------------------------------------------------------
gamma_n = 0.5
beta_n  = 0.25

# tiempo EXACTO del registro
t_total = float(t_ag[-1])
t = t_ag.copy()
ag_ext = ag_filt.copy()

# guardar para bloques siguientes
st.session_state["t_dyn"]   = t
st.session_state["ag_ext"]  = ag_ext
st.session_state["t_total"] = float(t_total)

# alturas (si hay nodes)
pisos_y = None
nodes = st.session_state.get("nodes", None)
try:
    if nodes is not None:
        arr = np.asarray(nodes, dtype=float)
        if arr.ndim >= 2 and arr.shape[1] >= 2:
            ys = arr[:, 1]
            pisos_y = np.sort(np.unique(ys[ys > 0]))
except Exception:
    pisos_y = None

# -----------------------------------------------------------------
# 📊 Layout simétrico
# -----------------------------------------------------------------
colL, colR = st.columns([1, 1], gap="large")

# ========================= IZQUIERDA (FIJA) =========================
with colL:
    with st.container(border=True):
        st.markdown(f"### {tr('b6_left_hdr')}")

        # Rayleigh FIJA con w del modelo fijo (evita w ~ 0)
        w_fix = st.session_state.get("w_sin", None)
        if w_fix is None:
            w_fix = modal_w(K_fix, M_fix)
        wR_fix = pick_two_w(w_fix, wmin=1e-6)

        alpha_fix, beta_fix = rayleigh_from_w(wR_fix, zeta)
        C_fix = alpha_fix * M_fix + beta_fix * K_fix

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("α", f"{alpha_fix:.3e}", "1/s")
        m2.metric("β", f"{beta_fix:.3e}", "s")
        m3.metric("ζ", f"{zeta:.3f}", "")
        m4.metric(tr("b6_metrics_dur"), f"{t_total:.2f}", "s")

        r_fix = np.ones((M_fix.shape[0], 1))
        P_fix = -(M_fix @ r_fix) @ ag_ext[np.newaxis, :]

        U0 = np.zeros(K_fix.shape[0])
        V0 = np.zeros(K_fix.shape[0])

        sig_fix = _sig(K_fix, M_fix, ag_ext, extra=(dt, zeta, gamma_n, beta_n, alpha_fix, beta_fix))
        cache_fix = st.session_state.get("b6_cache_fix", {})

        if cache_fix.get("sig") != sig_fix:
            u_fix, v_fix_t, a_fix_t = newmark(
                M_fix, C_fix, K_fix, U0, V0, dt, P_fix, gamma=gamma_n, beta=beta_n
            )
            u_fix, v_fix_t, a_fix_t = ensure_2d(u_fix, v_fix_t, a_fix_t)

            st.session_state["b6_cache_fix"] = {
                "sig": sig_fix,
                "t": t,
                "u": u_fix,
                "v": v_fix_t,
                "a": a_fix_t,
                "xlsx": None,
            }
        else:
            u_fix = cache_fix["u"]
            v_fix_t = cache_fix["v"]
            a_fix_t = cache_fix["a"]

        st.session_state["u_t"]   = u_fix
        st.session_state["v_t"]   = v_fix_t
        st.session_state["a_t"]   = a_fix_t
        st.session_state["t_fix"] = t
        st.session_state["C_fix"] = C_fix

        # PFA FIJA (a_abs = a_rel + ag)
        a_abs_fix = a_fix_t + ag_ext.reshape(1, -1)
        pfa_fix_mps2 = np.max(np.abs(a_abs_fix), axis=1)
        st.session_state["pfa_fix_mps2"] = np.asarray(pfa_fix_mps2, float).ravel()
        st.session_state["pfa_fix_g"]    = st.session_state["pfa_fix_mps2"] / 9.8066500000

        labels_fix = [f"Level_{i+1}" for i in range(int(u_fix.shape[0]))]

        if st.session_state["b6_cache_fix"].get("xlsx") is None:
            st.session_state["b6_cache_fix"]["xlsx"] = make_excel_per_floor(
                t=t, u=u_fix, v=v_fix_t, a=a_fix_t, sheet_names=labels_fix
            )

        st.download_button(
            label=f"⬇️ {tr('b6_dl_xlsx')}",
            data=st.session_state["b6_cache_fix"]["xlsx"],
            file_name="ISO-LATE_B6_FIXED.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help=tr("b6_dl_help"),
            use_container_width=True,
        )

    with st.container(border=True):
        st.markdown(f"### {tr('b6_left_resp')}")

        if pisos_y is None:
            pisos_y = np.arange(1, u_fix.shape[0] + 1, dtype=float)
        st.session_state["alturas"] = np.array(pisos_y, dtype=float)

        n_gdl_fix = int(u_fix.shape[0])
        for idx in range(n_gdl_fix):
            titulo = f"**{tr('b6_floor').format(i=idx+1)}**"
            with st.expander(titulo, expanded=(idx == 0)):
                _plot_resp(
                    t,
                    u_fix[[idx], :],
                    v_fix_t[[idx], :],
                    a_fix_t[[idx], :],
                    [float(pisos_y[idx]) if idx < len(pisos_y) else float(idx + 1)],
                    t_total,
                    nombre_piso=str(idx + 1),
                )

# ========================= DERECHA (AISLADA LINEAL - rigidez efectiva) =========================
with colR:
    with st.container(border=True):
        st.markdown(f"### {tr('b6_right_hdr')}")

        # Rayleigh AISLADA con modelo condensado (evita w ~ 0)
        w_ais = modal_w(K_ais_eff, M_ais_eff)
        wR_ais = pick_two_w(w_ais, wmin=1e-6)

        alpha_ais, beta_ais = rayleigh_from_w(wR_ais, zeta)
        C_ais = alpha_ais * M_ais_eff + beta_ais * K_ais_eff

        # número de aisladores
        n_aisladores = int(st.session_state.get("n_aisladores", 1))

        # ✅ amortiguamiento viscoso equivalente del aislador
        c_1ais = float(st.session_state["res_aislador"]["c_1ais"])
        C_ais[0, 0] += c_1ais * n_aisladores

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("α", f"{alpha_ais:.3e}", "1/s")
        m2.metric("β", f"{beta_ais:.3e}", "s")
        m3.metric("ζ", f"{zeta:.3f}", "")
        m4.metric(tr("b6_metrics_dur"), f"{t_total:.2f}", "s")

        sig_ais = _sig(K_ais_eff, M_ais_eff, C_ais, ag_ext, extra=(dt, zeta, gamma_n, beta_n))
        cache_ais = st.session_state.get("b6_cache_ais", {})

        if cache_ais.get("sig") != sig_ais:
            r_ais = np.ones((M_ais_eff.shape[0], 1))
            P_ais = -(M_ais_eff @ r_ais) @ ag_ext[np.newaxis, :]

            U0_ais = np.zeros(K_ais_eff.shape[0])
            V0_ais = np.zeros(K_ais_eff.shape[0])

            u_ais, v_ais_t, a_ais_t = newmark(
                M_ais_eff, C_ais, K_ais_eff, U0_ais, V0_ais, dt, P_ais, gamma=gamma_n, beta=beta_n
            )
            u_ais, v_ais_t, a_ais_t = ensure_2d(u_ais, v_ais_t, a_ais_t)

            st.session_state["b6_cache_ais"] = {
                "sig": sig_ais,
                "t": t,
                "u": u_ais,
                "v": v_ais_t,
                "a": a_ais_t,
                "xlsx": None,
            }
        else:
            u_ais = cache_ais["u"]
            v_ais_t = cache_ais["v"]
            a_ais_t = cache_ais["a"]

        # Guardar THA AISLADA completa
        st.session_state["u_t_ais"] = u_ais
        st.session_state["v_t_ais"] = v_ais_t
        st.session_state["a_t_ais"] = a_ais_t
        st.session_state["t_ais"]   = t
        st.session_state["C_ais"]   = C_ais

        # PFA AISLADA (a_abs = a_rel + ag)
        a_abs_ais = a_ais_t + ag_ext.reshape(1, -1)
        pfa_ais_mps2 = np.max(np.abs(a_abs_ais), axis=1)
        st.session_state["pfa_ais_mps2"] = np.asarray(pfa_ais_mps2, float).ravel()
        st.session_state["pfa_ais_g"]    = st.session_state["pfa_ais_mps2"] / 9.8066500000

        # PFA SOLO SUPERESTRUCTURA
        if a_ais_t.shape[0] >= 2:
            st.session_state["pfa_ais_super_mps2"] = st.session_state["pfa_ais_mps2"][1:].copy()
            st.session_state["pfa_ais_super_g"]    = st.session_state["pfa_ais_g"][1:].copy()
        else:
            st.session_state["pfa_ais_super_mps2"] = np.array([], dtype=float)
            st.session_state["pfa_ais_super_g"]    = np.array([], dtype=float)

        # Demanda aislador (GDL 0) — se conserva para otros bloques
        u_iso_t = np.asarray(u_ais[0, :], float).ravel()
        u_iso_max = float(np.max(np.abs(u_iso_t)))
        st.session_state["u_iso_t"]   = u_iso_t
        st.session_state["u_iso_max"] = float(u_iso_max)

        # Excel completo del sistema aislado (incluye aislador)
        labels_ais = ["Isolator_0"] + [f"Level_{i}" for i in range(1, int(u_ais.shape[0]))]

        if st.session_state["b6_cache_ais"].get("xlsx") is None:
            st.session_state["b6_cache_ais"]["xlsx"] = make_excel_per_floor(
                t=t, u=u_ais, v=v_ais_t, a=a_ais_t, sheet_names=labels_ais
            )

        st.download_button(
            label=f"⬇️ {tr('b6_dl_xlsx')}",
            data=st.session_state["b6_cache_ais"]["xlsx"],
            file_name="ISO-LATE_B6_ISOLATED.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help=tr("b6_dl_help"),
            use_container_width=True,
        )

    with st.container(border=True):
        st.markdown(f"### {tr('b6_right_resp')}")

        alt_fix = st.session_state.get("alturas", None)
        if alt_fix is None:
            # alturas de pisos reales solamente
            alt_fix = np.arange(1, max(u_ais.shape[0], 2), dtype=float)

        alt_fix = np.asarray(alt_fix, float).ravel()

        # se mantiene completo por compatibilidad con otros bloques
        st.session_state["alturas_ais"] = np.r_[0.0, alt_fix]

        # ✅ SOLO mostrar pisos reales, no el aislador
        n_gdl_ais = int(u_ais.shape[0])

        if n_gdl_ais <= 1:
            st.info("No hay pisos disponibles para mostrar en la superestructura.")
        else:
            for idx in range(1, n_gdl_ais):
                titulo = f"**{tr('b6_floor').format(i=idx)}**"
                nombre = str(idx)
                h = float(alt_fix[idx - 1]) if (idx - 1) < len(alt_fix) else float(idx)

                with st.expander(titulo, expanded=(idx == 1)):
                    _plot_resp(
                        t,
                        u_ais[[idx], :],
                        v_ais_t[[idx], :],
                        a_ais_t[[idx], :],
                        [h],
                        t_total,
                        nombre_piso=nombre,
                    )

st.success(tr("b6_ok"))

# =============================================================================
# === BLOQUE 7: ESPECTRO NEC24 (izq) + HISTÉRESIS (der) =======================
# =============================================================================
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# -------------------------------------------------------------------------
# ✅ Textos EN/ES (solo para este bloque) + HELPERS
# -------------------------------------------------------------------------
T["en"].update({
    "b7_title": "NEC-24 spectrum + isolator hysteresis",

    "b7_missing_vars": "Missing variables for this block: {keys}",
    "b7_missing_fun": "Function `newmark_nl_base_bilinear` was not found.",

    # Left side
    "b7_left_hdr": "NEC-24 inelastic spectrum + points (FIXED vs ISOLATED)",
    "b7_inelastic": "Inelastic spectrum (R={R:g})",
    "b7_fixed_mark": "FIXED: T₁={T:.3f}s | Sa={Sa:.3f}g",
    "b7_iso_mark": "ISOLATED: T₁={T:.3f}s | Sa={Sa:.3f}g",
    "b7_xlabel_T": "Period T [s]",
    "b7_ylabel_Sa": "Sa [g]",

    # Right side
    "b7_right_hdr": "Real isolator hysteresis (bilinear)",
    "b7_xlabel_u0": "Base displacement u₀ [m]",
    "b7_ylabel_Fiso": "Isolator force [Tf]",
})

T["es"].update({
    "b7_title": "Espectro NEC-24 + Histéresis del aislador",

    "b7_missing_vars": "⚠️ Faltan variables para este bloque: {keys}",
    "b7_missing_fun": "⚠️ No se encontró la función `newmark_nl_base_bilinear`.",

    # Left side
    "b7_left_hdr": "Espectro inelástico NEC-24 + puntos (FIJA vs AISLADA)",
    "b7_inelastic": "Espectro inelástico (R={R:g})",
    "b7_fixed_mark": "FIJA: T₁={T:.3f}s | Sa={Sa:.3f}g",
    "b7_iso_mark": "AISLADA: T₁={T:.3f}s | Sa={Sa:.3f}g",
    "b7_xlabel_T": "Período T [s]",
    "b7_ylabel_Sa": "Sa [g]",

    # Right side
    "b7_right_hdr": "Histéresis real del aislador (bilineal)",
    "b7_xlabel_u0": "Desplazamiento base u₀ [m]",
    "b7_ylabel_Fiso": "Fuerza aislador [Tf]",
})

# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
st.markdown(f"## 📌 {tr('b7_title')}")

# ----------------------- Estilos -----------------------
BG         = "#2B3141"
COLOR_TEXT = "#E8EDF2"
COLOR_GRID = "#5B657A"
HALO = [pe.withStroke(linewidth=2.4, foreground=BG), pe.Normal()]

COLOR_INELAST  = "#F2A6A0"
COLOR_MARK_FIX = "#FFE6A3"
COLOR_MARK_AIS = "#77DD77"
COLOR_GUIDE    = "#7A8498"
LEG_FACE       = "#3A4050"
LEG_EDGE       = "#A7B1C5"
COLOR_LINE1    = "#C79BFF"

# -----------------------------------------------------------------
# ✅ VALIDACIÓN DE VARIABLES NECESARIAS
# -----------------------------------------------------------------
required = [
    "rs_T_spec",
    "rs_Sa_inelas",
    "nec24_params",
    "T_sin",
    "T_ais",
    "M_cond_ais",
    "K_cond_ais",
    "dt",
    "ag_filt",
    "k_inicial_1ais",
    "k_post_1ais",
    "yield_1ais",
    "c_1ais",
]

missing = [k for k in required if k not in st.session_state]
if missing:
    st.warning(tr("b7_missing_vars").format(keys=", ".join(missing)))
    st.stop()

if "newmark_nl_base_bilinear" not in globals():
    st.warning(tr("b7_missing_fun"))
    st.stop()

# -----------------------------------------------------------------
# Layout
# -----------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")
FIG_W, FIG_H = 7.2, 4.8

# =============================================================================
# IZQUIERDA: ESPECTRO NEC INELÁSTICO + MARCADORES
# =============================================================================
with col_left:
    with st.container(border=True):
        st.subheader(f"📈 {tr('b7_left_hdr')}")

        T_plot  = np.asarray(st.session_state["rs_T_spec"], dtype=float).ravel()
        Sa_inel = np.asarray(st.session_state["rs_Sa_inelas"], dtype=float).ravel()

        R_spec  = float(st.session_state["nec24_params"]["R"])
        T_final = float(T_plot[-1])

        T1_fix = float(np.asarray(st.session_state["T_sin"], dtype=float).ravel()[0])
        T1_ais = float(np.asarray(st.session_state["T_ais"], dtype=float).ravel()[0])

        # ✅ Ambos puntos evaluados SOLO en el espectro inelástico
        Sa_Tfix = float(np.interp(T1_fix, T_plot, Sa_inel))
        Sa_Tais = float(np.interp(T1_ais, T_plot, Sa_inel))

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        ax.plot(
            T_plot, Sa_inel,
            lw=2.4,
            color=COLOR_INELAST,
            linestyle="--",
            label=tr("b7_inelastic").format(R=R_spec)
        )

        # FIJA
        ax.plot([T1_fix, T1_fix], [0, Sa_Tfix], color=COLOR_GUIDE, linestyle=":", lw=1.2)
        ax.plot([0, T1_fix], [Sa_Tfix, Sa_Tfix], color=COLOR_GUIDE, linestyle=":", lw=1.2)
        ax.plot(
            T1_fix, Sa_Tfix, "o",
            ms=7,
            mfc=COLOR_MARK_FIX,
            mec="none",
            label=tr("b7_fixed_mark").format(T=T1_fix, Sa=Sa_Tfix)
        )

        # AISLADA
        ax.plot([T1_ais, T1_ais], [0, Sa_Tais], color=COLOR_GUIDE, linestyle=":", lw=1.2)
        ax.plot([0, T1_ais], [Sa_Tais, Sa_Tais], color=COLOR_GUIDE, linestyle=":", lw=1.2)
        ax.plot(
            T1_ais, Sa_Tais, "o",
            ms=7,
            mfc=COLOR_MARK_AIS,
            mec="none",
            label=tr("b7_iso_mark").format(T=T1_ais, Sa=Sa_Tais)
        )

        ax.set_xlabel(tr("b7_xlabel_T"), color=COLOR_TEXT)
        ax.set_ylabel(tr("b7_ylabel_Sa"), color=COLOR_TEXT)
        ax.set_xlim(0, T_final)
        ax.set_ylim(bottom=0)

        ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
        ax.tick_params(colors=COLOR_TEXT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        leg = ax.legend(facecolor=LEG_FACE, edgecolor=LEG_EDGE, framealpha=0.95)
        for txt in leg.get_texts():
            txt.set_color(COLOR_TEXT)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

# =============================================================================
# DERECHA: HISTÉRESIS REAL (solo gráfico, sin nota)
# =============================================================================
with col_right:
    with st.container(border=True):
        st.subheader(f"🟣 {tr('b7_right_hdr')}")

        dt     = float(st.session_state["dt"])
        ag_g   = np.asarray(st.session_state["ag_filt"], dtype=float).ravel()

        M_ais  = np.array(st.session_state["M_cond_ais"], dtype=float)
        K_ais  = np.array(st.session_state["K_cond_ais"], dtype=float)

        # ✅ Mantengo la misma lógica: sin Rayleigh global aquí
        C_used = np.zeros_like(M_ais)

        k0    = float(st.session_state["k_inicial_1ais"])
        kp    = float(st.session_state["k_post_1ais"])
        Fy    = float(st.session_state["yield_1ais"])
        c_iso = float(st.session_state["c_1ais"])

        U_nl, V_nl, A_nl, Fiso_hist, Fhyst_hist, Ehyst = newmark_nl_base_bilinear(
            M=M_ais,
            C=C_used,
            K=K_ais,
            dt=dt,
            ag_g=ag_g,
            k0=k0,
            kp=kp,
            Fy=Fy,
            c_iso=c_iso,
            gamma=0.5,
            beta=0.25,
            newton_tol=1e-7,
            newton_maxit=30
        )

        st.session_state["U_nl"]      = U_nl
        st.session_state["Fiso_hist"] = Fiso_hist
        st.session_state["Ehyst"]     = Ehyst

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        ax.plot(U_nl[0, :], Fiso_hist, color=COLOR_LINE1, lw=0.3)

        ax.set_xlabel(tr("b7_xlabel_u0"), color=COLOR_TEXT)
        ax.set_ylabel(tr("b7_ylabel_Fiso"), color=COLOR_TEXT)

        ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.35)
        ax.tick_params(colors=COLOR_TEXT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

#st.success(tr("b7_ready"))

# =============================================================================
# === BLOQUE 8: CORTANTES POR PISO (RSA INELÁSTICO vs THA MAX/MIN) ============
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# -------------------------------------------------------------------------
# ✅ Textos EN/ES (solo para este bloque) + HELPERS
# -------------------------------------------------------------------------
T["en"].update({
    "b8_title": "Story shears – Inelastic response spectrum (RSA) & Time History (THA)",

    "b8_method": "Select shear-force method",
    "h_b8_method": "RSA: SRSS peak from the inelastic target spectrum. THA: time-history response using the real Max/Min envelope.",
    "b8_method_rsa": "Inelastic response spectrum analysis (RSA)",
    "b8_method_tha": "Time history analysis (THA)",

    "b8_need_alt": "Missing st.session_state['alturas'] (floor heights). Run Block 6 first.",
    "b8_need_nec": "Missing NEC spectrum from Block 3: rs_T_spec / rs_Sa_inelas.",
    "b8_spec_dim": "Spectrum arrays have inconsistent lengths.",
    "b8_need_fix_modal": "Missing FIXED: M_cond / v_norm_sin / T_sin.",
    "b8_need_iso_modal": "Missing ISOLATED: M_cond_ais / v_norm_ais / T_ais.",
    "b8_fix_dim_bad": "FIXED: v_norm_sin does not match number of DOFs.",
    "b8_iso_dim_bad": "ISOLATED: v_norm_ais does not match number of DOFs.",

    "b8_rsa_left": "FIXED – Inelastic RSA (SRSS)",
    "b8_rsa_right": "ISOLATED – Inelastic RSA (SRSS)",
    "b8_table_fix": "Show shear table (FIXED)",
    "b8_table_iso": "Show shear table (ISOLATED)",
    "b8_plot_fix_rsa": "Story shears – Inelastic RSA (FIXED) – ±SRSS envelope",
    "b8_plot_iso_rsa": "Story shears – Inelastic RSA (ISOLATED) – ±SRSS envelope",
    "b8_rsa_ok": "Story shears (inelastic RSA) ready.",

    "b8_need_tha": "Missing THA variables: {keys}",

    "b8_tha_fix": "FIXED – THA (Max/Min)",
    "b8_tha_iso": "ISOLATED – THA (Max/Min)",
    "b8_plot_fix_maxmin": "THA – Story shears Max/Min (FIXED)",
    "b8_plot_iso_maxmin": "THA – Story shears Max/Min (ISOLATED)",
    "b8_tha_ok": "Story shears (THA Max/Min) ready.",

    "b8_xlabel_V": "Shear V [tonf]",
    "b8_ylabel_h": "Height [m]",
})

T["es"].update({
    "b8_title": "Cortantes por piso – Espectro inelástico (RSA) y Tiempo historia (THA)",

    "b8_method": "Selecciona el método de cortantes",
    "h_b8_method": "RSA: SRSS desde el espectro inelástico. THA: tiempo-historia usando la envolvente real Max/Min.",
    "b8_method_rsa": "Análisis modal espectral inelástico (RSA)",
    "b8_method_tha": "Tiempo historia (THA)",

    "b8_need_alt": "❌ Falta st.session_state['alturas'] (alturas de pisos). Ejecuta el Bloque 6 primero.",
    "b8_need_nec": "❌ Falta el espectro NEC del Bloque 3: rs_T_spec / rs_Sa_inelas.",
    "b8_spec_dim": "❌ Espectro: dimensiones no coinciden.",
    "b8_need_fix_modal": "❌ Falta FIJA: M_cond / v_norm_sin / T_sin.",
    "b8_need_iso_modal": "❌ Falta AISLADA: M_cond_ais / v_norm_ais / T_ais.",
    "b8_fix_dim_bad": "❌ FIJA: v_norm_sin no coincide con n_dofs.",
    "b8_iso_dim_bad": "❌ AISLADA: v_norm_ais no coincide con n_dofs.",

    "b8_rsa_left": "🟦 FIJA – RSA inelástico (SRSS)",
    "b8_rsa_right": "🟩 AISLADA – RSA inelástico (SRSS)",
    "b8_table_fix": "📋 Ver tabla de cortantes (FIJA)",
    "b8_table_iso": "📋 Ver tabla de cortantes (AISLADA)",
    "b8_plot_fix_rsa": "Cortantes por piso – RSA inelástico (FIJA) – envolvente ±SRSS",
    "b8_plot_iso_rsa": "Cortantes por piso – RSA inelástico (AISLADA) – envolvente ±SRSS",
    "b8_rsa_ok": "✅ Cortantes por RSA inelástico listos",

    "b8_need_tha": "❌ Faltan variables THA: {keys}",

    "b8_tha_fix": "🟦 FIJA – THA (Max/Min)",
    "b8_tha_iso": "🟩 AISLADA – THA (Max/Min)",
    "b8_plot_fix_maxmin": "THA – Cortantes por piso Max/Min (FIJA)",
    "b8_plot_iso_maxmin": "THA – Cortantes por piso Max/Min (AISLADA)",
    "b8_tha_ok": "✅ Cortantes por THA Max/Min listos",

    "b8_xlabel_V": "Cortante V [tonf]",
    "b8_ylabel_h": "Altura [m]",
})

# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
st.markdown(f"## 🧱 {tr('b8_title')}")

# ----------------------- Estilos -----------------------
BG            = "#2B3141"
COLOR_TEXT    = "#E8EDF2"
COLOR_GRID    = "#5B657A"
COLOR_FIX     = "#A8D5FF"
COLOR_AIS     = "#77DD77"
COLOR_GUIDE   = "#FFDFA0"
HALO = [pe.withStroke(linewidth=2.4, foreground=BG), pe.Normal()]

# ----------------------- Helpers gráficos -----------------------
def _lw_by_n(n_pisos: int, lw_min=0.65, lw_max=2.2):
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

def _df_to_compact_table(df: pd.DataFrame, height_min=150, height_max=320):
    n = len(df)
    h = 44 + 26 * min(n, 9)
    h = int(max(height_min, min(height_max, h)))
    st.dataframe(df, hide_index=True, use_container_width=True, height=h)

def _etabs_polyline_xy(V_story, y_levels):
    V_story  = np.asarray(V_story, float).ravel()
    y_levels = np.asarray(y_levels, float).ravel()

    xs, ys = [], []
    xs.append(V_story[0]); ys.append(y_levels[0])
    for i in range(len(V_story)):
        xs += [V_story[i], V_story[i]]
        ys += [y_levels[i], y_levels[i+1]]
        if i < len(V_story) - 1:
            xs += [V_story[i+1]]
            ys += [y_levels[i+1]]
    return np.asarray(xs, float), np.asarray(ys, float)

def _plot_story_shear_etabs_maxmin(Vmax_story, Vmin_story, y_levels, title, color_line, nref):
    Vmax_story = np.asarray(Vmax_story, float).ravel()
    y_levels   = np.asarray(y_levels, float).ravel()
    n = len(Vmax_story)

    if len(y_levels) != n + 1:
        st.error(f"❌ y_levels debe ser n+1 y V_story n. V={Vmax_story.shape}, y={y_levels.shape}")
        return
    if n == 0:
        st.warning("⚠️ V_story vacío.")
        return

    lw = _lw_by_n(nref)
    ms = _ms_by_n(nref)

    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    xM, yM = _etabs_polyline_xy(Vmax_story, y_levels)
    ax.plot(xM, yM, "-", color=color_line, lw=lw)

    y_pts = y_levels
    x_pts_max = np.r_[Vmax_story[0], Vmax_story]
    x_pts_max = x_pts_max[:len(y_pts)]
    ax.plot(x_pts_max, y_pts, "o", color=color_line, ms=ms)

    if Vmin_story is not None:
        Vmin_story = np.asarray(Vmin_story, float).ravel()
        if len(Vmin_story) != n:
            st.error("❌ Vmin_story no coincide con Vmax_story.")
            return
        xm, ym = _etabs_polyline_xy(Vmin_story, y_levels)
        ax.plot(xm, ym, "-", color=color_line, lw=lw, alpha=0.90)

        x_pts_min = np.r_[Vmin_story[0], Vmin_story]
        x_pts_min = x_pts_min[:len(y_pts)]
        ax.plot(x_pts_min, y_pts, "o", color=color_line, ms=ms, alpha=0.85)

        vmax = float(np.max(np.abs(np.r_[Vmax_story, Vmin_story])))
    else:
        ax.plot(-xM, yM, "-", color=color_line, lw=lw, alpha=0.85)
        ax.plot(-x_pts_max, y_pts, "o", color=color_line, ms=ms, alpha=0.75)
        vmax = float(np.max(np.abs(Vmax_story)))

    ax.axvline(0.0, color=COLOR_GRID, lw=1.0, alpha=0.6)
    ax.set_xlabel(tr("b8_xlabel_V"), color=COLOR_TEXT)
    ax.set_ylabel(tr("b8_ylabel_h"), color=COLOR_TEXT)
    ax.set_title(title, color=COLOR_TEXT, fontweight="bold")
    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
    ax.tick_params(colors=COLOR_TEXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    vmax = 1.0 if (not np.isfinite(vmax) or vmax <= 0) else vmax
    ax.set_xlim(-1.12 * vmax, 1.12 * vmax)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# -------------------- Selector método --------------------
metodo = st.selectbox(
    tr("b8_method"),
    [tr("b8_method_rsa"), tr("b8_method_tha")],
    index=0,
    key="metodo_cortantes",
    help=tr("h_b8_method"),
)

# Alturas pisos
alt_fix = st.session_state.get("alturas", None)
if alt_fix is None:
    st.error(tr("b8_need_alt"))
    st.stop()

alt_fix = np.asarray(alt_fix, float).ravel()
n_pisos = int(len(alt_fix))
y_levels = np.r_[0.0, alt_fix]

# =============================================================================
# RSA (Modal espectral inelástico) – SUPER vs SUPER
# =============================================================================
if metodo == tr("b8_method_rsa"):

    T_spec = st.session_state.get("rs_T_spec", None)
    Sa_in  = st.session_state.get("rs_Sa_inelas", None)
    Ie     = float(st.session_state.get("rs_Ie", 1.0))

    if T_spec is None or Sa_in is None:
        st.error(tr("b8_need_nec"))
        st.stop()

    T_spec = np.asarray(T_spec, float).ravel()
    Sa_in  = np.asarray(Sa_in, float).ravel()

    if len(T_spec) != len(Sa_in):
        st.error(tr("b8_spec_dim"))
        st.stop()

    Sa_use = Sa_in * Ie
    g = 9.8066500000

    M_fix  = st.session_state.get("M_cond", None)
    Vn_fix = st.session_state.get("v_norm_sin", None)
    T_fix  = st.session_state.get("T_sin", None)
    if M_fix is None or Vn_fix is None or T_fix is None:
        st.error(tr("b8_need_fix_modal"))
        st.stop()

    M_fix  = np.asarray(M_fix, float)
    Vn_fix = np.asarray(Vn_fix, float)
    T_fix  = np.asarray(T_fix, float).ravel()
    n_fix  = int(M_fix.shape[0])

    M_ais  = st.session_state.get("M_cond_ais", st.session_state.get("M_cond_aislador", None))
    Vn_ais = st.session_state.get("v_norm_ais", None)
    T_ais  = st.session_state.get("T_ais", None)
    if M_ais is None or Vn_ais is None or T_ais is None:
        st.error(tr("b8_need_iso_modal"))
        st.stop()

    M_ais  = np.asarray(M_ais, float)
    Vn_ais = np.asarray(Vn_ais, float)
    T_ais  = np.asarray(T_ais, float).ravel()
    n_ais  = int(M_ais.shape[0])

    if Vn_fix.shape[0] != n_fix:
        st.error(tr("b8_fix_dim_bad"))
        st.stop()
    if Vn_ais.shape[0] != n_ais:
        st.error(tr("b8_iso_dim_bad"))
        st.stop()

    def _rsa_story_shear_super(Mmat, Vnorm, Tvec, n_pisos_ref):
        Mmat = np.asarray(Mmat, float)
        Vnorm = np.asarray(Vnorm, float)
        Tvec = np.asarray(Tvec, float).ravel()

        n = int(Mmat.shape[0])
        nmod = int(Vnorm.shape[1])
        has_iso = (n == (n_pisos_ref + 1))

        V_modes = []

        for rr in range(nmod):
            Tr = float(Tvec[rr]) if rr < len(Tvec) else np.nan
            if (not np.isfinite(Tr)) or Tr <= 0:
                continue

            Sa_r = float(np.interp(Tr, T_spec, Sa_use)) * g
            phi = np.asarray(Vnorm[:, rr], float).reshape(n, 1)

            r = np.ones((n, 1), float)

            num = float((phi.T @ Mmat @ r).item())
            den = float((phi.T @ Mmat @ phi).item())

            if (not np.isfinite(den)) or abs(den) < 1e-14:
                continue

            Gamma = num / den
            F_full = (Gamma * (Mmat @ phi) * Sa_r).ravel()

            if has_iso:
                F_use = F_full[1:1+n_pisos_ref]
            else:
                F_use = F_full[:n_pisos_ref]

            if len(F_use) != n_pisos_ref:
                return None

            V_r = np.zeros((n_pisos_ref,), float)
            for k in range(n_pisos_ref):
                V_r[k] = np.sum(F_use[k:])

            V_modes.append(V_r)

        if len(V_modes) == 0:
            return None

        V_modes = np.vstack(V_modes)
        return np.sqrt(np.sum(V_modes**2, axis=0))

    V_fix_srss = _rsa_story_shear_super(M_fix, Vn_fix, T_fix, n_pisos)
    V_ais_srss = _rsa_story_shear_super(M_ais, Vn_ais, T_ais, n_pisos)

    if V_fix_srss is None:
        st.error("❌ RSA FIJA: no se pudieron armar modos válidos o dimensiones no calzan.")
        st.stop()
    if V_ais_srss is None:
        st.error("❌ RSA AISLADA: no se pudieron armar modos válidos o dimensiones no calzan.")
        st.stop()

    V_fix_srss = np.asarray(V_fix_srss, float).ravel()
    V_ais_srss = np.asarray(V_ais_srss, float).ravel()

    V_fix_max, V_fix_min = V_fix_srss.copy(), -V_fix_srss.copy()
    V_ais_max, V_ais_min = V_ais_srss.copy(), -V_ais_srss.copy()

    df_fix = pd.DataFrame({
        "Piso": np.arange(1, n_pisos + 1),
        "Altura sup [m]": np.round(alt_fix, 3),
        "Vmax [tonf]": np.round(V_fix_max, 6),
        "Vmin [tonf]": np.round(V_fix_min, 6),
        "|V|max [tonf]": np.round(np.maximum(np.abs(V_fix_max), np.abs(V_fix_min)), 6),
    })
    df_ais = pd.DataFrame({
        "Piso": np.arange(1, n_pisos + 1),
        "Altura sup [m]": np.round(alt_fix, 3),
        "Vmax [tonf]": np.round(V_ais_max, 6),
        "Vmin [tonf]": np.round(V_ais_min, 6),
        "|V|max [tonf]": np.round(np.maximum(np.abs(V_ais_max), np.abs(V_ais_min)), 6),
    })

    colL, colR = st.columns([1, 1], gap="large")

    with colL:
        with st.container(border=True):
            st.subheader(tr("b8_rsa_left"))
            with st.expander(tr("b8_table_fix"), expanded=False):
                _df_to_compact_table(df_fix)
            _plot_story_shear_etabs_maxmin(
                V_fix_max, None, y_levels, tr("b8_plot_fix_rsa"), COLOR_FIX, nref=n_pisos
            )

    with colR:
        with st.container(border=True):
            st.subheader(tr("b8_rsa_right"))
            with st.expander(tr("b8_table_iso"), expanded=False):
                _df_to_compact_table(df_ais)
            _plot_story_shear_etabs_maxmin(
                V_ais_max, None, y_levels, tr("b8_plot_iso_rsa"), COLOR_AIS, nref=n_pisos
            )

    st.success(tr("b8_rsa_ok"))

    st.session_state["cmp_V_fix_story"]     = np.maximum(np.abs(V_fix_max), np.abs(V_fix_min))
    st.session_state["cmp_V_ais_story"]     = np.maximum(np.abs(V_ais_max), np.abs(V_ais_min))
    st.session_state["cmp_V_fix_story_max"] = V_fix_max
    st.session_state["cmp_V_fix_story_min"] = V_fix_min
    st.session_state["cmp_V_ais_story_max"] = V_ais_max
    st.session_state["cmp_V_ais_story_min"] = V_ais_min
    st.session_state["cmp_tag_shear"] = (
        "Inelastic RSA (SRSS) ±" if st.session_state.get("lang", "en") == "en"
        else "RSA inelástico (SRSS) ±"
    )
    st.session_state["cmp_Vb_fix"]          = float(V_fix_max[0]) if len(V_fix_max) else np.nan
    st.session_state["cmp_Vb_ais"]          = float(V_ais_max[0]) if len(V_ais_max) else np.nan

# =============================================================================
# THA (Tiempo historia) – SOLO Max/Min
# =============================================================================
else:
    dt = st.session_state.get("dt", None)
    ag = st.session_state.get("ag_filt", None)

    a_fix = st.session_state.get("a_t", None)
    M_fix = st.session_state.get("M_cond", None)

    a_ais = st.session_state.get("a_t_ais", None)
    M_ais = st.session_state.get("M_cond_ais", st.session_state.get("M_cond_aislador", None))

    falt = []
    if dt is None: falt.append("dt")
    if ag is None: falt.append("ag_filt")
    if a_fix is None: falt.append("a_t")
    if M_fix is None: falt.append("M_cond")
    if a_ais is None: falt.append("a_t_ais")
    if M_ais is None: falt.append("M_cond_ais (o M_cond_aislador)")
    if falt:
        st.error(tr("b8_need_tha").format(keys=", ".join(falt)))
        st.stop()

    dt = float(dt)
    ag = np.asarray(ag, float).ravel()

    a_fix = np.asarray(a_fix, float); a_fix = a_fix if a_fix.ndim == 2 else a_fix[np.newaxis, :]
    a_ais = np.asarray(a_ais, float); a_ais = a_ais if a_ais.ndim == 2 else a_ais[np.newaxis, :]

    def _match_ag(ag_in, nt):
        ag2 = np.asarray(ag_in, float).ravel()
        if len(ag2) < nt:
            ag2 = np.pad(ag2, (0, nt - len(ag2)), mode="constant")
        else:
            ag2 = ag2[:nt]
        t = np.arange(nt, dtype=float) * dt
        return t, ag2

    def _floor_forces(Mmat, a_rel, ag_series):
        a_rel = np.asarray(a_rel, float)
        n, nt = a_rel.shape
        m = np.diag(np.asarray(Mmat, float)).reshape(n, 1)
        a_abs = a_rel + ag_series.reshape(1, nt)
        return m * a_abs

    def _story_from_forces(F):
        F = np.asarray(F, float)
        n, nt = F.shape
        V = np.zeros_like(F)
        for k in range(n):
            V[k, :] = np.sum(F[k:, :], axis=0)
        return V

    t_fix, ag_fix = _match_ag(ag, a_fix.shape[1])
    F_fix = _floor_forces(M_fix, a_fix, ag_fix)
    if F_fix.shape[0] != n_pisos:
        st.error("❌ THA FIJA: fuerzas no calzan con n_pisos (revisa GDL cond).")
        st.stop()
    V_fix_all = _story_from_forces(F_fix)

    t_ais, ag_ais = _match_ag(ag, a_ais.shape[1])

    if a_ais.shape[0] == n_pisos + 1:
        a_base_rel = a_ais[0:1, :]
        a_sup_rel_base = a_ais[1:1+n_pisos, :] - a_base_rel

        m_sup = np.diag(np.asarray(M_ais, float))[1:1+n_pisos].reshape(n_pisos, 1)
        F_ais_sup = m_sup * a_sup_rel_base
        V_ais_all = _story_from_forces(F_ais_sup)

    elif a_ais.shape[0] == n_pisos:
        m_sup = np.diag(np.asarray(M_ais, float)).reshape(n_pisos, 1)
        F_ais_sup = m_sup * a_ais
        V_ais_all = _story_from_forces(F_ais_sup)

    else:
        st.error("❌ THA AISLADA: dimensiones no calzan con n_pisos ni n_pisos+1.")
        st.stop()

    V_fix_max = np.max(V_fix_all, axis=1)
    V_fix_min = np.min(V_fix_all, axis=1)
    V_ais_max = np.max(V_ais_all, axis=1)
    V_ais_min = np.min(V_ais_all, axis=1)

    V_fix_cmp = np.maximum(np.abs(V_fix_max), np.abs(V_fix_min))
    V_ais_cmp = np.maximum(np.abs(V_ais_max), np.abs(V_ais_min))

    title_fix = tr("b8_plot_fix_maxmin")
    title_ais = tr("b8_plot_iso_maxmin")

    V_fix_max = np.asarray(V_fix_max, float).ravel()
    V_fix_min = np.asarray(V_fix_min, float).ravel()
    V_ais_max = np.asarray(V_ais_max, float).ravel()
    V_ais_min = np.asarray(V_ais_min, float).ravel()

    colL, colR = st.columns([1, 1], gap="large")

    with colL:
        with st.container(border=True):
            st.subheader(tr("b8_tha_fix"))

            df = pd.DataFrame({
                "Piso": np.arange(1, n_pisos + 1),
                "Altura sup [m]": np.round(alt_fix, 3),
                "Vmax [tonf]": np.round(V_fix_max, 6),
                "Vmin [tonf]": np.round(V_fix_min, 6),
                "|V|max [tonf]": np.round(np.maximum(np.abs(V_fix_max), np.abs(V_fix_min)), 6),
            })

            with st.expander(tr("b8_table_fix"), expanded=False):
                _df_to_compact_table(df)

            _plot_story_shear_etabs_maxmin(V_fix_max, V_fix_min, y_levels, title_fix, COLOR_FIX, nref=n_pisos)

    with colR:
        with st.container(border=True):
            st.subheader(tr("b8_tha_iso"))

            df = pd.DataFrame({
                "Piso": np.arange(1, n_pisos + 1),
                "Altura sup [m]": np.round(alt_fix, 3),
                "Vmax [tonf]": np.round(V_ais_max, 6),
                "Vmin [tonf]": np.round(V_ais_min, 6),
                "|V|max [tonf]": np.round(np.maximum(np.abs(V_ais_max), np.abs(V_ais_min)), 6),
            })

            with st.expander(tr("b8_table_iso"), expanded=False):
                _df_to_compact_table(df)

            _plot_story_shear_etabs_maxmin(V_ais_max, V_ais_min, y_levels, title_ais, COLOR_AIS, nref=n_pisos)

    st.success(tr("b8_tha_ok"))

    st.session_state["cmp_V_fix_story"]     = np.asarray(V_fix_cmp, float).ravel()
    st.session_state["cmp_V_ais_story"]     = np.asarray(V_ais_cmp, float).ravel()
    st.session_state["cmp_V_fix_story_max"] = V_fix_max
    st.session_state["cmp_V_ais_story_max"] = V_ais_max
    st.session_state["cmp_V_fix_story_min"] = V_fix_min
    st.session_state["cmp_V_ais_story_min"] = V_ais_min
    st.session_state["cmp_tag_shear"]       = "THA (Max/Min)"
    st.session_state["cmp_Vb_fix"]          = float(st.session_state["cmp_V_fix_story"][0]) if len(st.session_state["cmp_V_fix_story"]) else np.nan
    st.session_state["cmp_Vb_ais"]          = float(st.session_state["cmp_V_ais_story"][0]) if len(st.session_state["cmp_V_ais_story"]) else np.nan
    
# =============================================================================
# === BLOQUE 9: DESPLAZAMIENTOS LATERALES (RSA INELÁSTICO vs THA MAX/MIN) =====
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# -------------------------------------------------------------------------
# ✅ Textos EN/ES (solo para este bloque) + HELPERS
# -------------------------------------------------------------------------
T["en"].update({
    "b9_title": "Lateral displacements – Inelastic response spectrum (RSA) & Time History (THA)",
    "b9_method": "Select displacement method",
    "h_b9_method": "RSA: peak displacements from the inelastic target spectrum (SRSS). THA: real Max/Min envelope over time.",
    "b9_method_rsa": "Inelastic response spectrum analysis (RSA)",
    "b9_method_tha": "Time history analysis (THA)",
    "b9_need_alt": "Missing st.session_state['alturas'] (floor heights). Run Block 6 first.",
    "b9_need_spec": "Missing spectrum from Block 3: rs_T_spec / rs_Sa_inelas.",
    "b9_spec_dim": "Spectrum arrays have inconsistent lengths: T={t} Sa_inel={ine}.",
    "b9_need_fix": "Missing FIXED: M_cond / v_norm_sin / T_sin.",
    "b9_need_iso": "Missing ISOLATED: M_cond_ais / v_norm_ais / T_ais.",
    "b9_fix_dim": "FIXED: v_norm_sin rows ({r}) do not match n_dofs ({n}).",
    "b9_iso_dim": "ISOLATED: v_norm_ais rows ({r}) do not match n_dofs ({n}).",
    "b9_rsa_fix_hdr": "FIXED – Inelastic RSA (SRSS)",
    "b9_rsa_iso_hdr": "ISOLATED – Inelastic RSA (SRSS)",
    "b9_tbl_fix": "Show table (FIXED)",
    "b9_tbl_iso": "Show table (ISOLATED)",
    "b9_plot_fix_rsa": "Displacement profile – Inelastic RSA (FIXED)",
    "b9_plot_iso_rsa": "Displacement profile – Inelastic RSA (ISOLATED)",
    "b9_rsa_ok": "Inelastic RSA ready – FIXED vs ISOLATED.",
    "b9_need_tha": "Missing THA variables: {keys}",
    "b9_tha_fix_hdr": "FIXED – THA (Max/Min)",
    "b9_tha_iso_hdr": "ISOLATED – THA (Max/Min)",
    "b9_tha_tag_maxmin": "max/min envelope",
    "b9_plot_fix_maxmin": "THA – Displacement profile Max/Min (FIXED)",
    "b9_plot_iso_maxmin": "THA – Displacement profile Max/Min (ISOLATED)",
    "b9_tha_ok": "THA ready – FIXED vs ISOLATED.",
    "b9_xlabel": "Displacement u [m]",
    "b9_ylabel": "Height [m]",
})

T["es"].update({
    "b9_title": "Desplazamientos laterales – Espectro inelástico (RSA) y Tiempo historia (THA)",
    "b9_method": "Selecciona el método de desplazamientos",
    "h_b9_method": "RSA: picos desde el espectro inelástico (SRSS). THA: envolvente real Max/Min en el tiempo.",
    "b9_method_rsa": "Análisis modal espectral inelástico (RSA)",
    "b9_method_tha": "Tiempo historia (THA)",
    "b9_need_alt": "❌ Falta st.session_state['alturas'] (alturas de pisos). Ejecuta el Bloque 6 primero.",
    "b9_need_spec": "❌ Falta el espectro del Bloque 3: rs_T_spec / rs_Sa_inelas.",
    "b9_spec_dim": "❌ Espectro: dimensiones no coinciden: T={t} Sa_inel={ine}.",
    "b9_need_fix": "❌ Falta FIJA: M_cond / v_norm_sin / T_sin.",
    "b9_need_iso": "❌ Falta AISLADA: M_cond_ais / v_norm_ais / T_ais.",
    "b9_fix_dim": "❌ FIJA: v_norm_sin filas ({r}) != n_dofs ({n}).",
    "b9_iso_dim": "❌ AISLADA: v_norm_ais filas ({r}) != n_dofs ({n}).",
    "b9_rsa_fix_hdr": "🟦 FIJA – RSA inelástico (SRSS)",
    "b9_rsa_iso_hdr": "🟩 AISLADA – RSA inelástico (SRSS)",
    "b9_tbl_fix": "📋 Ver tabla (FIJA)",
    "b9_tbl_iso": "📋 Ver tabla (AISLADA)",
    "b9_plot_fix_rsa": "Perfil de desplazamientos – RSA inelástico (FIJA)",
    "b9_plot_iso_rsa": "Perfil de desplazamientos – RSA inelástico (AISLADA)",
    "b9_rsa_ok": "✅ RSA inelástico listo – FIJA vs AISLADA.",
    "b9_need_tha": "❌ Faltan variables THA: {keys}",
    "b9_tha_fix_hdr": "🟦 FIJA – THA (Max/Min)",
    "b9_tha_iso_hdr": "🟩 AISLADA – THA (Max/Min)",
    "b9_tha_tag_maxmin": "envolvente max/min",
    "b9_plot_fix_maxmin": "THA – Perfil de desplazamientos Max/Min (FIJA)",
    "b9_plot_iso_maxmin": "THA – Perfil de desplazamientos Max/Min (AISLADA)",
    "b9_tha_ok": "✅ THA listo – FIJA vs AISLADA.",
    "b9_xlabel": "Desplazamiento u [m]",
    "b9_ylabel": "Altura [m]",
})

st.markdown(f"## 🟦 {tr('b9_title')}")

BG            = "#2B3141"
COLOR_TEXT    = "#E8EDF2"
COLOR_GRID    = "#5B657A"
COLOR_FIX     = "#FFE6A3"
COLOR_AIS     = "#77DD77"
HALO = [pe.withStroke(linewidth=2.4, foreground=BG), pe.Normal()]

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

def _plot_profile(U, y, title, color_line, nref, xlabel_key="b9_xlabel"):
    U = np.asarray(U, float).ravel()
    y = np.asarray(y, float).ravel()
    lw = _lw_by_n(nref)
    ms = _ms_by_n(nref)

    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(U, y, "-o", color=color_line, lw=lw, ms=ms)

    ax.axvline(0.0, color=COLOR_GRID, lw=1.0, alpha=0.6)
    ax.set_xlabel(tr(xlabel_key), color=COLOR_TEXT)
    ax.set_ylabel(tr("b9_ylabel"), color=COLOR_TEXT)
    ax.set_title(title, color=COLOR_TEXT, fontweight="bold")
    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
    ax.tick_params(colors=COLOR_TEXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

def _plot_profile_maxmin(Umax, Umin, y, title, color_line, nref, xlabel_key="b9_xlabel"):
    Umax = np.asarray(Umax, float).ravel()
    Umin = np.asarray(Umin, float).ravel()
    y    = np.asarray(y, float).ravel()

    if (len(Umax) != len(y)) or (len(Umin) != len(y)):
        st.error(f"❌ Max/Min no calza con niveles. Umax={Umax.shape}, Umin={Umin.shape}, y={y.shape}")
        return

    lw = _lw_by_n(nref)
    ms = _ms_by_n(nref)

    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(Umax, y, "-o", color=color_line, lw=lw, ms=ms, alpha=0.95)
    ax.plot(Umin, y, "-o", color=color_line, lw=lw, ms=ms, alpha=0.75)

    ax.axvline(0.0, color=COLOR_GRID, lw=1.0, alpha=0.6)
    ax.set_xlabel(tr(xlabel_key), color=COLOR_TEXT)
    ax.set_ylabel(tr("b9_ylabel"), color=COLOR_TEXT)
    ax.set_title(title, color=COLOR_TEXT, fontweight="bold")
    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
    ax.tick_params(colors=COLOR_TEXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    vmax = float(np.max(np.abs(np.r_[Umax, Umin])))
    vmax = 1.0 if (not np.isfinite(vmax) or vmax <= 0) else vmax
    ax.set_xlim(-1.12 * vmax, 1.12 * vmax)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

metodo = st.selectbox(
    tr("b9_method"),
    [tr("b9_method_rsa"), tr("b9_method_tha")],
    index=0,
    key="metodo_desplazamientos",
    help=tr("h_b9_method"),
)

alt_fix = st.session_state.get("alturas", None)
if alt_fix is None:
    st.error(tr("b9_need_alt"))
    st.stop()

alt_fix = np.asarray(alt_fix, float).ravel()
n_pisos = int(len(alt_fix))
y_levels = np.r_[0.0, alt_fix]

# =============================================================================
# RSA (Modal espectral inelástico)
# =============================================================================
if metodo == tr("b9_method_rsa"):

    T_spec = st.session_state.get("rs_T_spec", None)
    Sa_in  = st.session_state.get("rs_Sa_inelas", None)
    Ie     = float(st.session_state.get("rs_Ie", 1.0))

    if T_spec is None or Sa_in is None:
        st.error(tr("b9_need_spec"))
        st.stop()

    T_spec = np.asarray(T_spec, float).ravel()
    Sa_in  = np.asarray(Sa_in, float).ravel()

    if len(T_spec) != len(Sa_in):
        st.error(tr("b9_spec_dim").format(t=T_spec.shape, ine=Sa_in.shape))
        st.stop()

    Sa_use = Sa_in * Ie

    M_fix  = st.session_state.get("M_cond", None)
    Vn_fix = st.session_state.get("v_norm_sin", None)
    T_fix  = st.session_state.get("T_sin", None)
    w_fix  = st.session_state.get("w_sin", None)
    if M_fix is None or Vn_fix is None or T_fix is None:
        st.error(tr("b9_need_fix"))
        st.stop()

    M_fix  = np.asarray(M_fix, float)
    Vn_fix = np.asarray(Vn_fix, float)
    T_fix  = np.asarray(T_fix, float).ravel()
    n_fix  = int(M_fix.shape[0])

    M_ais  = st.session_state.get("M_cond_ais", st.session_state.get("M_cond_aislador", None))
    Vn_ais = st.session_state.get("v_norm_ais", None)
    T_ais  = st.session_state.get("T_ais", None)
    w_ais  = st.session_state.get("w_ais", None)
    if M_ais is None or Vn_ais is None or T_ais is None:
        st.error(tr("b9_need_iso"))
        st.stop()

    M_ais  = np.asarray(M_ais, float)
    Vn_ais = np.asarray(Vn_ais, float)
    T_ais  = np.asarray(T_ais, float).ravel()
    n_ais  = int(M_ais.shape[0])

    if Vn_fix.shape[0] != n_fix:
        st.error(tr("b9_fix_dim").format(r=Vn_fix.shape[0], n=n_fix))
        st.stop()
    if Vn_ais.shape[0] != n_ais:
        st.error(tr("b9_iso_dim").format(r=Vn_ais.shape[0], n=n_ais))
        st.stop()

    g = 9.8066500000

    def _srss_u_fixed(Mmat, Vnorm, Tvec, wvec):
        n = Mmat.shape[0]
        r = np.ones((n, 1), float)

        U_modes = []
        nmod = Vnorm.shape[1]
        wvec_use = None if wvec is None else np.asarray(wvec, float).ravel()

        for rr in range(nmod):
            Tr = float(Tvec[rr]) if rr < len(Tvec) else np.nan
            if (not np.isfinite(Tr)) or Tr <= 0:
                continue

            Sa_r = float(np.interp(Tr, T_spec, Sa_use)) * g
            phi  = Vnorm[:, rr].reshape(n, 1)

            Mr = float((phi.T @ Mmat @ r).item())
            Mm = float((phi.T @ Mmat @ phi).item())
            if (not np.isfinite(Mm)) or abs(Mm) < 1e-18:
                continue
            Gamma = Mr / Mm

            if (wvec_use is not None) and (rr < len(wvec_use)) and np.isfinite(wvec_use[rr]) and (wvec_use[rr] > 0):
                wv = float(wvec_use[rr])
            else:
                wv = 2.0 * np.pi / Tr
            if (not np.isfinite(wv)) or wv <= 0:
                continue

            q_r = Gamma * Sa_r / (wv**2)
            U_modes.append((phi * q_r).ravel())

        if len(U_modes) == 0:
            return None
        U_modes = np.vstack(U_modes)
        return np.sqrt(np.sum(U_modes**2, axis=0))

    def _srss_u_isolated_abs_levels(Mmat, Vnorm, Tvec, wvec, n_pisos_target):
        n = Mmat.shape[0]
        r = np.ones((n, 1), float)

        U_modes = []
        nmod = Vnorm.shape[1]
        wvec_use = None if wvec is None else np.asarray(wvec, float).ravel()

        for rr in range(nmod):
            Tr = float(Tvec[rr]) if rr < len(Tvec) else np.nan
            if (not np.isfinite(Tr)) or Tr <= 0:
                continue

            Sa_r = float(np.interp(Tr, T_spec, Sa_use)) * g
            phi  = Vnorm[:, rr].reshape(n, 1)

            Mr = float((phi.T @ Mmat @ r).item())
            Mm = float((phi.T @ Mmat @ phi).item())
            if (not np.isfinite(Mm)) or abs(Mm) < 1e-18:
                continue
            Gamma = Mr / Mm

            if (wvec_use is not None) and (rr < len(wvec_use)) and np.isfinite(wvec_use[rr]) and (wvec_use[rr] > 0):
                wv = float(wvec_use[rr])
            else:
                wv = 2.0 * np.pi / Tr
            if (not np.isfinite(wv)) or wv <= 0:
                continue

            q_r = Gamma * Sa_r / (wv**2)
            U_modes.append((phi * q_r).ravel())

        if len(U_modes) == 0:
            return None

        U_modes = np.vstack(U_modes)
        u_srss = np.sqrt(np.sum(U_modes**2, axis=0))

        if n == n_pisos_target + 1:
            return u_srss
        elif n == n_pisos_target:
            return np.r_[0.0, u_srss]
        else:
            return None

    u_fix_srss = _srss_u_fixed(M_fix, Vn_fix, T_fix, w_fix)
    if u_fix_srss is None:
        st.error("❌ RSA FIJA: no pude armar modos válidos (revisa T_sin / w_sin / v_norm_sin).")
        st.stop()
    if len(u_fix_srss) != n_pisos:
        st.error(f"❌ RSA FIJA: u={u_fix_srss.shape} no coincide con n_pisos={n_pisos}.")
        st.stop()

    u_ais_abs_levels = _srss_u_isolated_abs_levels(M_ais, Vn_ais, T_ais, w_ais, n_pisos)
    if u_ais_abs_levels is None:
        st.error("❌ RSA AISLADA: no pude formar SRSS ABS. Revisa v_norm_ais / T_ais / M_cond_ais.")
        st.stop()

    u_fix_plot = np.r_[0.0, np.asarray(u_fix_srss, float).ravel()]
    u_ais_plot = np.asarray(u_ais_abs_levels, float).ravel()

    niv = np.arange(0, n_pisos + 1)
    h_tab = np.r_[0.0, alt_fix]

    dfL = pd.DataFrame({
        "Nivel": niv,
        "Altura [m]": np.round(h_tab, 3),
        "u_rsa [m]": np.round(u_fix_plot, 6),
        "u_rsa [mm]": np.round(u_fix_plot * 1000.0, 3),
    })
    dfR = pd.DataFrame({
        "Nivel": niv,
        "Altura [m]": np.round(h_tab, 3),
        "u_rsa_abs [m]": np.round(u_ais_plot, 6),
        "u_rsa_abs [mm]": np.round(u_ais_plot * 1000.0, 3),
    })

    colL, colR = st.columns([1, 1], gap="large")
    with colL:
        with st.container(border=True):
            st.subheader(tr("b9_rsa_fix_hdr"))
            with st.expander(tr("b9_tbl_fix"), expanded=False):
                _df_to_compact_table(dfL)
            _plot_profile(u_fix_plot, y_levels, tr("b9_plot_fix_rsa"), COLOR_FIX, nref=n_pisos)

    with colR:
        with st.container(border=True):
            st.subheader(tr("b9_rsa_iso_hdr"))
            with st.expander(tr("b9_tbl_iso"), expanded=False):
                _df_to_compact_table(dfR)
            _plot_profile(u_ais_plot, y_levels, tr("b9_plot_iso_rsa"), COLOR_AIS, nref=n_pisos)

    st.success(tr("b9_rsa_ok"))

    st.session_state["cmp_U_fix_levels"] = np.asarray(u_fix_plot, float).ravel()
    st.session_state["cmp_U_ais_levels"] = np.asarray(u_ais_plot, float).ravel()
    st.session_state["cmp_tag_disp"] = (
        "Inelastic RSA (SRSS)" if st.session_state.get("lang", "en") == "en"
        else "RSA inelástico (SRSS)"
    )

# =============================================================================
# THA (solo Max/Min)
# =============================================================================
else:
    dt    = st.session_state.get("dt", None)
    u_fix = st.session_state.get("u_t", None)
    u_ais = st.session_state.get("u_t_ais", None)

    falt = []
    if dt is None: falt.append("dt")
    if u_fix is None: falt.append("u_t")
    if u_ais is None: falt.append("u_t_ais")
    if falt:
        st.error(tr("b9_need_tha").format(keys=", ".join(falt)))
        st.stop()

    dt = float(dt)
    u_fix = np.asarray(u_fix, float); u_fix = u_fix if u_fix.ndim == 2 else u_fix[np.newaxis, :]
    u_ais = np.asarray(u_ais, float); u_ais = u_ais if u_ais.ndim == 2 else u_ais[np.newaxis, :]

    if u_fix.shape[0] != n_pisos:
        st.error(f"❌ THA FIJA: u_fix={u_fix.shape} no calza con n_pisos={n_pisos}.")
        st.stop()

    if u_ais.shape[0] == n_pisos + 1:
        u_ais_use = u_ais
    elif u_ais.shape[0] == n_pisos:
        u_ais_use = np.vstack([np.zeros((1, u_ais.shape[1])), u_ais])
    else:
        st.error(f"❌ THA AISLADA: u_ais={u_ais.shape} no calza con n_pisos ni n_pisos+1.")
        st.stop()

    U_fix_max = np.max(u_fix, axis=1)
    U_fix_min = np.min(u_fix, axis=1)
    U_ais_max = np.max(u_ais_use, axis=1)
    U_ais_min = np.min(u_ais_use, axis=1)

    U_fix_plot_max = np.r_[0.0, U_fix_max]
    U_fix_plot_min = np.r_[0.0, U_fix_min]

    st.session_state["cmp_U_fix_levels"] = np.maximum(np.abs(U_fix_plot_max), np.abs(U_fix_plot_min))
    st.session_state["cmp_U_ais_levels"] = np.maximum(np.abs(U_ais_max), np.abs(U_ais_min))
    st.session_state["cmp_tag_disp"]     = f"THA ({tr('b9_tha_tag_maxmin')})"

    # -----------------------------------------------------------------
    # NUEVO: historias completas por nivel para derivas THA exactas
    # -----------------------------------------------------------------
    nt_fix = u_fix.shape[1]
    U_fix_hist_levels = np.vstack([np.zeros((1, nt_fix)), u_fix])  # base fija = 0
    
    st.session_state["cmp_U_fix_hist_levels"] = np.asarray(U_fix_hist_levels, float)
    st.session_state["cmp_U_ais_hist_levels"] = np.asarray(u_ais_use, float)

    niv = np.arange(0, n_pisos + 1)
    h_tab = np.r_[0.0, alt_fix]

    dfL = pd.DataFrame({
        "Nivel": niv,
        "Altura [m]": np.round(h_tab, 3),
        "u_max [m]": np.round(U_fix_plot_max, 6),
        "u_min [m]": np.round(U_fix_plot_min, 6),
        "|u|max [m]": np.round(np.maximum(np.abs(U_fix_plot_max), np.abs(U_fix_plot_min)), 6),
    })
    dfR = pd.DataFrame({
        "Nivel": niv,
        "Altura [m]": np.round(h_tab, 3),
        "u_max_abs [m]": np.round(U_ais_max, 6),
        "u_min_abs [m]": np.round(U_ais_min, 6),
        "|u|max [m]": np.round(np.maximum(np.abs(U_ais_max), np.abs(U_ais_min)), 6),
    })

    colL, colR = st.columns([1, 1], gap="large")
    with colL:
        with st.container(border=True):
            st.subheader(tr("b9_tha_fix_hdr").format(tag=tr("b9_tha_tag_maxmin")))
            with st.expander(tr("b9_tbl_fix"), expanded=False):
                _df_to_compact_table(dfL)
            _plot_profile_maxmin(
                U_fix_plot_max, U_fix_plot_min, y_levels,
                tr("b9_plot_fix_maxmin"), COLOR_FIX, nref=n_pisos
            )

    with colR:
        with st.container(border=True):
            st.subheader(tr("b9_tha_iso_hdr").format(tag=tr("b9_tha_tag_maxmin")))
            with st.expander(tr("b9_tbl_iso"), expanded=False):
                _df_to_compact_table(dfR)
            _plot_profile_maxmin(
                U_ais_max, U_ais_min, y_levels,
                tr("b9_plot_iso_maxmin"), COLOR_AIS, nref=n_pisos
            )

    st.success(tr("b9_tha_ok"))

# =============================================================================
# === BLOQUE 10: DERIVAS NEC24 (RSA INELÁSTICO vs THA MAX/MIN) =================
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# -------------------------------------------------------------------------
# ✅ Textos EN/ES (solo para este bloque) + HELPERS
# -------------------------------------------------------------------------
T["en"].update({
    "b10_title": "NEC-24 story drifts – Inelastic RSA & THA",

    "b10_need_b9": "Missing output from Block 9. Needed: cmp_U_fix_levels, cmp_U_ais_levels, cmp_tag_disp.",
    "b10_src": "Source: **{tag}** (from displacements)",

    "b10_need_alt": "Missing st.session_state['alturas'] (floor heights).",
    "b10_dim_fix": "cmp_U_fix_levels={u} does not match expected levels n_pisos={n}.",
    "b10_dim_iso": "cmp_U_ais_levels={u} does not match expected levels n_pisos={n}.",

    "b10_nec_cd": "Cd (NEC-24)",
    "h_b10_nec_cd": "Deflection amplification factor Cd according to NEC-24.",
    "b10_need_Ie": "I cannot find Ie in session_state. Need rs_Ie or nec24_params['Ie'].",
    "b10_bad_Ie": "Invalid Ie (<=0).",
    "b10_nec_caption": "NEC drift = (Cd·real drift)/I with Cd={cd:.3g} and I={ie:.3g}.",

    "b10_err_calc": "Error computing drifts: {e}",

    "b10_story": "Story",
    "b10_hsup": "Top height [m]",
    "b10_drift": "NEC drift",
    "b10_drift_pct": "NEC drift [%]",

    "b10_fix_hdr": "FIXED – {tag}",
    "b10_iso_hdr": "ISOLATED – {tag}",

    "b10_tbl_fix": "Show drift table (FIXED)",
    "b10_tbl_iso": "Show drift table (ISOLATED)",

    "b10_plot_fix": "Maximum NEC-24 story drifts – FIXED",
    "b10_plot_iso": "Maximum NEC-24 story drifts – ISOLATED",

    "b10_xlabel_nec": "NEC-24 drift [%]",
    "b10_ylabel": "Top height [m]",

    "b10_ok": "NEC-24 drifts ready",
    "b10_base": "Base",
})

T["es"].update({
    "b10_title": "Derivas NEC24 – RSA inelástico y THA",

    "b10_need_b9": "❌ Falta salida del BLOQUE 9. Necesito: cmp_U_fix_levels, cmp_U_ais_levels, cmp_tag_disp.",
    "b10_src": "🔗 Fuente: **{tag}** (desde desplazamientos)",

    "b10_need_alt": "❌ Falta st.session_state['alturas'] (alturas de pisos).",
    "b10_dim_fix": "❌ cmp_U_fix_levels={u} no coincide con niveles esperados n_pisos={n}.",
    "b10_dim_iso": "❌ cmp_U_ais_levels={u} no coincide con niveles esperados n_pisos={n}.",

    "b10_nec_cd": "Cd (NEC24)",
    "h_b10_nec_cd": "Factor de amplificación de desplazamientos Cd según NEC-24.",
    "b10_need_Ie": "❌ No encuentro Ie en session_state. Necesito rs_Ie o nec24_params['Ie'].",
    "b10_bad_Ie": "❌ Ie inválido (<=0).",
    "b10_nec_caption": "✅ Deriva_NEC = (Cd·Deriva_real)/I con Cd={cd:.3g} e I={ie:.3g}.",

    "b10_err_calc": "❌ Error calculando derivas: {e}",

    "b10_story": "Entrepiso",
    "b10_hsup": "Altura sup [m]",
    "b10_drift": "Deriva NEC24",
    "b10_drift_pct": "Deriva NEC24 [%]",

    "b10_fix_hdr": "🟦 FIJA – {tag}",
    "b10_iso_hdr": "🟩 AISLADA – {tag}",

    "b10_tbl_fix": "📋 Ver tabla de derivas (FIJA)",
    "b10_tbl_iso": "📋 Ver tabla de derivas (AISLADA)",

    "b10_plot_fix": "Máximas derivas NEC24 por entrepiso – FIJA",
    "b10_plot_iso": "Máximas derivas NEC24 por entrepiso – AISLADA",

    "b10_xlabel_nec": "Deriva NEC24 [%]",
    "b10_ylabel": "Altura sup [m]",

    "b10_ok": "Derivas NEC24 listas",
    "b10_base": "Base",
})

# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
st.markdown(f"## 📐 {tr('b10_title')}")

# ----------------------- Estilos -----------------------
BG            = "#2B3141"
COLOR_TEXT    = "#E8EDF2"
COLOR_GRID    = "#5B657A"
COLOR_FIX     = "#FFE6A3"
COLOR_AIS     = "#77DD77"
HALO = [pe.withStroke(linewidth=2.4, foreground=BG), pe.Normal()]

# ----------------------- Helpers -----------------------
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

def _calc_drifts_fixed_from_levels_abs(u_levels, y_levels):
    """
    Derivas FIJAS:
      drift_i = |u_i - u_{i-1}| / (y_i - y_{i-1})

    u_levels y y_levels deben ser (n_pisos+1,) con base incluida.
    Retorna:
      drift_stories (n_pisos,) asociado a alturas superiores y_story = y[1:]
    """
    u = np.asarray(u_levels, float).ravel()
    y = np.asarray(y_levels, float).ravel()

    if len(u) != len(y):
        raise ValueError(f"u_levels y y_levels deben tener igual tamaño. u={u.shape}, y={y.shape}")

    du = np.diff(u)
    dh = np.diff(y)

    if np.any(dh <= 1e-12):
        raise ValueError("Hay incrementos de altura dh<=0. Revisa 'alturas'.")

    drift = np.abs(du) / dh
    y_story = y[1:]
    return drift, y_story

def _calc_drifts_isolated_from_abs_via_iso(u_levels_abs, y_levels):
    """
    Derivas AISLADAS:
      1) convertir desplazamientos absolutos a relativos al aislador:
           u_rel = u_abs - u_iso
      2) usar:
           nivel 0 relativo = 0
           drift_1 = |u_rel_1 - 0| / h1
           drift_2 = |u_rel_2 - u_rel_1| / h2
           ...
      donde:
           h1 = y1 - 0
           h2 = y2 - y1
           ...

    Si u_abs = [u_iso, u1, u2, ..., un]
    y y      = [0,    y1, y2, ..., yn]

    retorna:
      drift (n_pisos,)
      y_story = [y1, y2, ..., yn]
    """
    u = np.asarray(u_levels_abs, float).ravel()
    y = np.asarray(y_levels, float).ravel()

    if len(u) != len(y):
        raise ValueError(f"u_levels y y_levels deben tener igual tamaño. u={u.shape}, y={y.shape}")

    n_pisos = len(y) - 1
    if n_pisos < 1:
        raise ValueError("No hay pisos para calcular derivas.")

    u_iso = float(u[0])
    u_rel_floors = u[1:] - u_iso  # tamaño n_pisos

    u_rel_levels = np.r_[0.0, u_rel_floors]  # base relativa al aislador = 0
    du = np.diff(u_rel_levels)
    dh = np.diff(y)

    if np.any(dh <= 1e-12):
        raise ValueError("Hay incrementos de altura dh<=0. Revisa 'alturas'.")

    drift = np.abs(du) / dh
    y_story = y[1:]
    return drift, y_story

def _plot_drift_poly(drift_plot, y_plot, title, color_line, xlabel, n_pisos_ref=10):
    """
    Plot tipo ETABS: polilínea con puntos.
    drift_plot y y_plot deben tener igual tamaño.
    El gráfico se muestra en porcentaje.
    """
    x = np.asarray(drift_plot, float).ravel() * 100.0
    y = np.asarray(y_plot, float).ravel()

    if len(x) != len(y):
        st.error(f"❌ drift_plot y y_plot deben tener igual tamaño. x={x.shape}, y={y.shape}")
        return

    lw = _lw_by_n(n_pisos_ref)
    ms = _ms_by_n(n_pisos_ref)

    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(x, y, "-o", color=color_line, lw=lw, ms=ms)

    ax.axvline(0.0, color=COLOR_GRID, lw=1.0, alpha=0.6)
    ax.set_xlabel(xlabel, color=COLOR_TEXT)
    ax.set_ylabel(tr("b10_ylabel"), color=COLOR_TEXT)
    ax.set_title(title, color=COLOR_TEXT, fontweight="bold")
    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
    ax.tick_params(colors=COLOR_TEXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    vmax = float(np.max(x)) if len(x) else 1.0
    vmax = 1.0 if vmax <= 0 else vmax
    ax.set_xlim(0.0, 1.10 * vmax)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# -------------------- HEREDAR desde BLOQUE 9 --------------------
u_fix_levels = st.session_state.get("cmp_U_fix_levels", None)  # (n_pisos+1,) base=0 + pisos
u_ais_levels = st.session_state.get("cmp_U_ais_levels", None)  # (n_pisos+1,) aislador + pisos
tag_disp     = st.session_state.get("cmp_tag_disp", None)

if u_fix_levels is None or u_ais_levels is None or tag_disp is None:
    st.error(tr("b10_need_b9"))
    st.stop()

st.caption(tr("b10_src").format(tag=tag_disp))

# -------------------- Alturas --------------------
alt_fix = st.session_state.get("alturas", None)
if alt_fix is None:
    st.error(tr("b10_need_alt"))
    st.stop()

alt_fix = np.asarray(alt_fix, float).ravel()
n_pisos = int(len(alt_fix))

# niveles: Base(0 m) + elevaciones de pisos
y_levels = np.r_[0.0, alt_fix]
expect_levels = n_pisos + 1

u_fix_levels = np.asarray(u_fix_levels, float).ravel()
u_ais_levels = np.asarray(u_ais_levels, float).ravel()

if len(u_fix_levels) != expect_levels:
    st.error(tr("b10_dim_fix").format(u=u_fix_levels.shape, n=n_pisos))
    st.stop()
if len(u_ais_levels) != expect_levels:
    st.error(tr("b10_dim_iso").format(u=u_ais_levels.shape, n=n_pisos))
    st.stop()

# -------------------------------------------------------------------------
# NEC24: SIEMPRE activo
# -------------------------------------------------------------------------
Cd = st.number_input(
    tr("b10_nec_cd"),
    0.1, 20.0, 5.5, 0.1,
    key="nec_Cd_derivas",
    help=tr("h_b10_nec_cd"),
)

Ie = st.session_state.get("rs_Ie", None)
if Ie is None:
    nec_params = st.session_state.get("nec24_params", {})
    Ie = nec_params.get("Ie", None)

if Ie is None:
    st.error(tr("b10_need_Ie"))
    st.stop()

Ie = float(Ie)
if Ie <= 0:
    st.error(tr("b10_bad_Ie"))
    st.stop()

st.caption(tr("b10_nec_caption").format(cd=float(Cd), ie=float(Ie)))

# =============================================================================
# Helpers THA exactos
# =============================================================================
def _calc_drifts_fixed_from_hist(u_hist_levels, y_levels):
    """
    FIJA THA:
      u_hist_levels: (n_pisos+1, nt) con base incluida en la fila 0
      drift_i(t) = |u_i(t) - u_{i-1}(t)| / h_i

    retorna:
      drift_max (n_pisos,)
      y_story = y[1:]
    """
    U = np.asarray(u_hist_levels, float)
    y = np.asarray(y_levels, float).ravel()

    if U.ndim != 2:
        raise ValueError(f"u_hist_levels debe ser 2D. Llegó {U.shape}")
    if U.shape[0] != len(y):
        raise ValueError(f"Filas de U={U.shape[0]} no coinciden con niveles={len(y)}")

    du_t = np.diff(U, axis=0)
    dh = np.diff(y).reshape(-1, 1)

    if np.any(dh <= 1e-12):
        raise ValueError("Hay incrementos de altura dh<=0. Revisa 'alturas'.")

    drift_t = np.abs(du_t) / dh
    drift_max = np.max(drift_t, axis=1)
    y_story = y[1:]
    return drift_max, y_story


def _calc_drifts_isolated_from_hist(u_hist_levels, y_levels):
    """
    AISLADA THA:
      u_hist_levels: (n_pisos+1, nt) con:
        fila 0 = u_aislador(t)
        fila 1 = u_piso1(t)
        fila 2 = u_piso2(t)
        ...

      drift_1(t) = |u_1(t) - u_iso(t)| / h1
      drift_2(t) = |u_2(t) - u_1(t)| / h2
      ...

    retorna:
      drift_max (n_pisos,)
      y_story = y[1:]
    """
    U = np.asarray(u_hist_levels, float)
    y = np.asarray(y_levels, float).ravel()

    if U.ndim != 2:
        raise ValueError(f"u_hist_levels debe ser 2D. Llegó {U.shape}")
    if U.shape[0] != len(y):
        raise ValueError(f"Filas de U={U.shape[0]} no coinciden con niveles={len(y)}")

    du_t = np.diff(U, axis=0)
    dh = np.diff(y).reshape(-1, 1)

    if np.any(dh <= 1e-12):
        raise ValueError("Hay incrementos de altura dh<=0. Revisa 'alturas'.")

    drift_t = np.abs(du_t) / dh
    drift_max = np.max(drift_t, axis=1)
    y_story = y[1:]
    return drift_max, y_story


# =============================================================================
# Cálculo de derivas reales base
# =============================================================================
try:
    if "THA" in str(tag_disp):
        u_fix_hist = st.session_state.get("cmp_U_fix_hist_levels", None)
        u_ais_hist = st.session_state.get("cmp_U_ais_hist_levels", None)

        if u_fix_hist is None or u_ais_hist is None:
            raise ValueError("Faltan historias THA guardadas desde el Bloque 9.")

        drift_fix_real, y_story_fix = _calc_drifts_fixed_from_hist(u_fix_hist, y_levels)
        drift_ais_real, y_story_ais = _calc_drifts_isolated_from_hist(u_ais_hist, y_levels)

    else:
        drift_fix_real, y_story_fix = _calc_drifts_fixed_from_levels_abs(u_fix_levels, y_levels)
        drift_ais_real, y_story_ais = _calc_drifts_isolated_from_abs_via_iso(u_ais_levels, y_levels)

except Exception as e:
    st.error(tr("b10_err_calc").format(e=e))
    st.stop()

# =============================================================================
# Aplicar NEC24
# =============================================================================
drift_fix = (float(Cd) * drift_fix_real) / float(Ie)
drift_ais = (float(Cd) * drift_ais_real) / float(Ie)
xlabel_plot = tr("b10_xlabel_nec")

# =============================================================================
# Para plots y tablas
# =============================================================================
y_plot_fix = np.r_[0.0, y_story_fix]
drift_fix_plot = np.r_[0.0, drift_fix]

y_plot_ais = np.r_[0.0, y_story_ais]
drift_ais_plot = np.r_[0.0, drift_ais]

# =============================================================================
# Tablas
# =============================================================================
entrep_tbl_fix = [tr("b10_base")] + ["0→1"] + [f"{i}→{i+1}" for i in range(1, len(drift_fix))]
dfL = pd.DataFrame({
    tr("b10_story"): entrep_tbl_fix,
    tr("b10_hsup"): np.round(y_plot_fix, 3),
    tr("b10_drift"): np.round(drift_fix_plot, 6),
    tr("b10_drift_pct"): np.round(drift_fix_plot * 100.0, 3),
})

entrep_tbl_ais = [tr("b10_base")] + ["0→1"] + [f"{i}→{i+1}" for i in range(1, len(drift_ais))]
dfR = pd.DataFrame({
    tr("b10_story"): entrep_tbl_ais,
    tr("b10_hsup"): np.round(y_plot_ais, 3),
    tr("b10_drift"): np.round(drift_ais_plot, 6),
    tr("b10_drift_pct"): np.round(drift_ais_plot * 100.0, 3),
})

# =============================================================================
# Layout
# =============================================================================
colL, colR = st.columns([1, 1], gap="large")

with colL:
    with st.container(border=True):
        st.subheader(tr("b10_fix_hdr").format(tag=tag_disp))
        with st.expander(tr("b10_tbl_fix"), expanded=False):
            _df_to_compact_table(dfL)
        _plot_drift_poly(
            drift_fix_plot, y_plot_fix,
            tr("b10_plot_fix"), COLOR_FIX, xlabel_plot, n_pisos_ref=n_pisos
        )

with colR:
    with st.container(border=True):
        st.subheader(tr("b10_iso_hdr").format(tag=tag_disp))
        with st.expander(tr("b10_tbl_iso"), expanded=False):
            _df_to_compact_table(dfR)
        _plot_drift_poly(
            drift_ais_plot, y_plot_ais,
            tr("b10_plot_iso"), COLOR_AIS, xlabel_plot, n_pisos_ref=n_pisos
        )

# Guardar para Bloque 11
st.session_state["cmp_drift_fix"] = np.asarray(drift_fix, float).ravel()
st.session_state["cmp_drift_ais"] = np.asarray(drift_ais, float).ravel()
st.session_state["cmp_tag_drift"] = (
    "NEC-24" if st.session_state.get("lang", "en") == "en"
    else "NEC24"
)

st.session_state["cmp_drift_fix_levels"] = np.asarray(drift_fix_plot, float).ravel()
st.session_state["cmp_drift_ais_levels"] = np.asarray(drift_ais_plot, float).ravel()
st.session_state["cmp_drift_y_fix_levels"] = np.asarray(y_plot_fix, float).ravel()
st.session_state["cmp_drift_y_ais_levels"] = np.asarray(y_plot_ais, float).ravel()

# compatibilidad con bloques previos
st.session_state["cmp_drift_y_levels"] = np.asarray(y_plot_fix, float).ravel()

st.success(tr("b10_ok"))

# =============================================================================
# === BLOQUE 11: COMPARATIVO FINAL – FIJA vs AISLADA (SIN CONTROLES) ==========
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# -------------------------------------------------------------------------
# ✅ Textos EN/ES (solo para este bloque)
#    Consistentes con Bloques 8, 9 y 10:
#    - Cortantes: tonf
#    - Desplazamientos: m
#    - Derivas: NEC24 [%]
# -------------------------------------------------------------------------
T["en"].update({
    "b11_title": "Comparison – FIXED vs ISOLATED",

    "b11_need": "Missing variables for this block: {keys}",
    "b11_tag_shear": "Shear source: **{tag}**",
    "b11_tag_disp": "Displacement source: **{tag}**",
    "b11_tag_drift": "Drift source: **{tag}**",

    "b11_hdr_V": "Story shears (FIXED vs ISOLATED)",
    "b11_hdr_U": "Lateral displacements (FIXED vs ISOLATED)",
    "b11_hdr_D": "Story drifts (FIXED vs ISOLATED)",
    "b11_hdr_S": "Final assessment",

    "b11_xlabel_V": "Shear V [tonf]",
    "b11_xlabel_U": "Displacement u [m]",
    "b11_xlabel_D": "NEC-24 drift [%]",
    "b11_ylabel_h": "Height [m]",

    "b11_fix": "FIXED",
    "b11_ais": "ISOLATED",

    "b11_help_v": "Base shear V₀ at level 0.",
    "b11_help_u": "Roof displacement u(H): peak roof displacement.",
    "b11_help_d": "Maximum NEC-24 interstory drift.",
    "b11_help_pfa": "Maximum absolute floor acceleration in g (a_abs = a_rel + ag).",
    "b11_help_lT": "λT: ratio between the 1st-mode period of isolated vs fixed models.",
    "b11_help_eV": "ηV: ratio between isolated and fixed base shear (smaller is typically better).",
    "b11_help_use": "Isolator use: demand/capacity at the isolator DOF (u_max/u_cap).",
    "b11_help_u_iso_max": "Peak absolute isolator displacement demand (DOF 0) from time-history: u_iso_max = max|u_iso(t)|.",

    "b11_u_iso_max": "u_iso_max [m]",
    "b11_vbase": "Base shear V₀",
    "b11_roof_u": "Roof displacement u(H)",
    "b11_drift_max": "Max NEC-24 drift",
    "b11_pfa_max": "Max PFA (abs) [g]",
    "b11_lambdaT": "λT = T_iso/T_fix (mode 1)",
    "b11_etaV": "ηV = Vb_iso/Vb_fix",
    "b11_iso_use": "Isolator use (u_max/u_cap)",

    "b11_ok": "Comparison ready.",
})

T["es"].update({
    "b11_title": "Comparativo – FIJA vs AISLADA",

    "b11_need": "❌ Faltan variables para este bloque: {keys}",
    "b11_tag_shear": "Fuente cortantes: **{tag}**",
    "b11_tag_disp": "Fuente desplazamientos: **{tag}**",
    "b11_tag_drift": "Fuente derivas: **{tag}**",

    "b11_hdr_V": "Cortantes por piso (FIJA vs AISLADA)",
    "b11_hdr_U": "Desplazamientos laterales (FIJA vs AISLADA)",
    "b11_hdr_D": "Derivas por entrepiso (FIJA vs AISLADA)",
    "b11_hdr_S": "Evaluación final",

    "b11_xlabel_V": "Cortante V [tonf]",
    "b11_xlabel_U": "Desplazamiento u [m]",
    "b11_xlabel_D": "Deriva NEC24 [%]",
    "b11_ylabel_h": "Altura [m]",

    "b11_fix": "FIJA",
    "b11_ais": "AISLADA",

    "b11_help_v": "Cortante basal V₀ en el nivel 0.",
    "b11_help_u": "Desplazamiento en techo u(H): desplazamiento máximo del último nivel.",
    "b11_help_d": "Deriva máxima NEC24 por entrepiso.",
    "b11_help_pfa": "Máxima aceleración absoluta de piso en g (a_abs = a_rel + ag).",
    "b11_help_lT": "λT: relación entre el periodo del modo 1 aislado y el fijo.",
    "b11_help_eV": "ηV: relación entre el cortante basal aislado y el fijo (menor suele ser mejor).",
    "b11_help_use": "Uso del aislador: demanda/capacidad en el GDL del aislador (u_max/u_cap).",
    "b11_help_u_iso_max": "Demanda máxima absoluta de desplazamiento del aislador (GDL 0) desde tiempo-historia: u_iso_max = max|u_iso(t)|.",

    "b11_u_iso_max": "u_iso_max [m]",
    "b11_vbase": "Cortante basal V₀",
    "b11_roof_u": "Desplazamiento en techo u(H)",
    "b11_drift_max": "Deriva máxima NEC24",
    "b11_pfa_max": "PFA máx (abs) [g]",
    "b11_lambdaT": "λT = T_iso/T_fix (modo 1)",
    "b11_etaV": "ηV = Vb_iso/Vb_fix",
    "b11_iso_use": "Uso del aislador (u_max/u_cap)",

    "b11_ok": "Comparativo listo.",
})

# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
st.markdown(f"## 📊 {tr('b11_title')}")

# ----------------------- Estilos -----------------------
BG         = "#2B3141"
COLOR_TEXT = "#E8EDF2"
COLOR_GRID = "#5B657A"
COLOR_FIX  = "#A8D5FF"   # consistente con Bloque 8 para fija
COLOR_AIS  = "#77DD77"   # consistente en todos los bloques
HALO = [pe.withStroke(linewidth=2.4, foreground=BG), pe.Normal()]

# ----------------------- Helpers -----------------------
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

def _etabs_polyline_xy(V_story, y_levels):
    V_story  = np.asarray(V_story, float).ravel()
    y_levels = np.asarray(y_levels, float).ravel()

    xs, ys = [], []
    xs.append(V_story[0]); ys.append(y_levels[0])
    for i in range(len(V_story)):
        xs += [V_story[i], V_story[i]]
        ys += [y_levels[i], y_levels[i+1]]
        if i < len(V_story) - 1:
            xs += [V_story[i+1]]
            ys += [y_levels[i+1]]
    return np.asarray(xs, float), np.asarray(ys, float)

def _plot_story_shear_compare(V_fix_max, V_fix_min, V_ais_max, V_ais_min, y_levels, title, n_pisos_ref, mode_tag):
    V_fix_max = np.asarray(V_fix_max, float).ravel()
    V_ais_max = np.asarray(V_ais_max, float).ravel()
    y_levels  = np.asarray(y_levels, float).ravel()

    n = len(V_fix_max)
    if len(V_ais_max) != n or len(y_levels) != n + 1:
        st.error("❌ Error de dimensiones en cortantes del comparativo final.")
        return

    lw = _lw_by_n(n_pisos_ref)
    ms = _ms_by_n(n_pisos_ref)

    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # FIJA
    x1, y1 = _etabs_polyline_xy(V_fix_max, y_levels)
    ax.plot(x1, y1, "-", color=COLOR_FIX, lw=lw, label=tr("b11_fix"))
    ax.plot(np.r_[V_fix_max[0], V_fix_max], y_levels, "o", color=COLOR_FIX, ms=ms)

    # AISLADA
    x2, y2 = _etabs_polyline_xy(V_ais_max, y_levels)
    ax.plot(x2, y2, "-", color=COLOR_AIS, lw=lw, label=tr("b11_ais"))
    ax.plot(np.r_[V_ais_max[0], V_ais_max], y_levels, "o", color=COLOR_AIS, ms=ms)

    # Si hay THA Max/Min, mostrar ambos; si no, reflejar para RSA signless
    if V_fix_min is not None and V_ais_min is not None:
        V_fix_min = np.asarray(V_fix_min, float).ravel()
        V_ais_min = np.asarray(V_ais_min, float).ravel()

        x1m, y1m = _etabs_polyline_xy(V_fix_min, y_levels)
        ax.plot(x1m, y1m, "-", color=COLOR_FIX, lw=lw, alpha=0.75)
        ax.plot(np.r_[V_fix_min[0], V_fix_min], y_levels, "o", color=COLOR_FIX, ms=ms, alpha=0.75)

        x2m, y2m = _etabs_polyline_xy(V_ais_min, y_levels)
        ax.plot(x2m, y2m, "-", color=COLOR_AIS, lw=lw, alpha=0.75)
        ax.plot(np.r_[V_ais_min[0], V_ais_min], y_levels, "o", color=COLOR_AIS, ms=ms, alpha=0.75)

        vmax = float(np.max(np.abs(np.r_[V_fix_max, V_fix_min, V_ais_max, V_ais_min])))
    else:
        ax.plot(-x1, y1, "-", color=COLOR_FIX, lw=lw, alpha=0.55)
        ax.plot(-np.r_[V_fix_max[0], V_fix_max], y_levels, "o", color=COLOR_FIX, ms=ms, alpha=0.55)

        ax.plot(-x2, y2, "-", color=COLOR_AIS, lw=lw, alpha=0.55)
        ax.plot(-np.r_[V_ais_max[0], V_ais_max], y_levels, "o", color=COLOR_AIS, ms=ms, alpha=0.55)

        vmax = float(np.max(np.abs(np.r_[V_fix_max, V_ais_max])))

    ax.axvline(0.0, color=COLOR_GRID, lw=1.0, alpha=0.6)
    ax.set_xlabel(tr("b11_xlabel_V"), color=COLOR_TEXT)
    ax.set_ylabel(tr("b11_ylabel_h"), color=COLOR_TEXT)
    ax.set_title(title, color=COLOR_TEXT, fontweight="bold")
    if mode_tag:
        ax.text(0.98, 0.02, str(mode_tag), transform=ax.transAxes,
                ha="right", va="bottom", color=COLOR_TEXT, fontsize=9, alpha=0.85)
    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
    ax.tick_params(colors=COLOR_TEXT)
    ax.legend(facecolor=BG, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, loc="best")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    vmax = 1.0 if (not np.isfinite(vmax) or vmax <= 0) else vmax
    ax.set_xlim(-1.12 * vmax, 1.12 * vmax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

def _plot_profile_compare(U_fix_max, U_fix_min, U_ais_max, U_ais_min, y_levels, title, n_pisos_ref, xlabel, mode_tag):
    U_fix_max = np.asarray(U_fix_max, float).ravel()
    U_ais_max = np.asarray(U_ais_max, float).ravel()
    y_levels  = np.asarray(y_levels, float).ravel()

    if len(U_fix_max) != len(y_levels) or len(U_ais_max) != len(y_levels):
        st.error("❌ Error de dimensiones en desplazamientos del comparativo final.")
        return

    lw = _lw_by_n(n_pisos_ref)
    ms = _ms_by_n(n_pisos_ref)

    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # --------------------------------------------------------------
    # FIJA
    # --------------------------------------------------------------
    ax.plot(U_fix_max, y_levels, "-o", color=COLOR_FIX, lw=lw, ms=ms, label=tr("b11_fix"))

    # --------------------------------------------------------------
    # AISLADA
    # --------------------------------------------------------------
    ax.plot(U_ais_max, y_levels, "-o", color=COLOR_AIS, lw=lw, ms=ms, label=tr("b11_ais"))

    # --------------------------------------------------------------
    # THA: usar max/min reales si existen
    # RSA: reflejar simétricamente el perfil positivo
    # --------------------------------------------------------------
    if U_fix_min is not None and U_ais_min is not None:
        U_fix_min = np.asarray(U_fix_min, float).ravel()
        U_ais_min = np.asarray(U_ais_min, float).ravel()

        ax.plot(U_fix_min, y_levels, "-o", color=COLOR_FIX, lw=lw, ms=ms, alpha=0.75)
        ax.plot(U_ais_min, y_levels, "-o", color=COLOR_AIS, lw=lw, ms=ms, alpha=0.75)

        vmax = float(np.max(np.abs(np.r_[U_fix_max, U_fix_min, U_ais_max, U_ais_min])))

    else:
        # RSA o caso sin min explícito: reflejo simétrico
        ax.plot(-U_fix_max, y_levels, "-o", color=COLOR_FIX, lw=lw, ms=ms, alpha=0.60)
        ax.plot(-U_ais_max, y_levels, "-o", color=COLOR_AIS, lw=lw, ms=ms, alpha=0.60)

        vmax = float(np.max(np.abs(np.r_[U_fix_max, U_ais_max])))

    ax.axvline(0.0, color=COLOR_GRID, lw=1.0, alpha=0.6)
    ax.set_xlabel(xlabel, color=COLOR_TEXT)
    ax.set_ylabel(tr("b11_ylabel_h"), color=COLOR_TEXT)
    ax.set_title(title, color=COLOR_TEXT, fontweight="bold")

    if mode_tag:
        ax.text(
            0.98, 0.02, str(mode_tag),
            transform=ax.transAxes,
            ha="right", va="bottom",
            color=COLOR_TEXT, fontsize=9, alpha=0.85
        )

    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
    ax.tick_params(colors=COLOR_TEXT)
    ax.legend(facecolor=BG, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, loc="best")

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    vmax = 1.0 if (not np.isfinite(vmax) or vmax <= 0) else vmax
    ax.set_xlim(-1.12 * vmax, 1.12 * vmax)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

def _plot_drift_compare_positive(D_fix_levels, D_ais_levels, y_levels, title, n_pisos_ref, xlabel, mode_tag):
    D_fix_levels = np.asarray(D_fix_levels, float).ravel()
    D_ais_levels = np.asarray(D_ais_levels, float).ravel()
    y_levels     = np.asarray(y_levels, float).ravel()

    if len(D_fix_levels) != len(y_levels) or len(D_ais_levels) != len(y_levels):
        st.error("❌ Error de dimensiones en derivas del comparativo final.")
        return

    lw = _lw_by_n(n_pisos_ref)
    ms = _ms_by_n(n_pisos_ref)

    x_fix = D_fix_levels * 100.0
    x_ais = D_ais_levels * 100.0

    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(x_fix, y_levels, "-o", color=COLOR_FIX, lw=lw, ms=ms, label=tr("b11_fix"))
    ax.plot(x_ais, y_levels, "-o", color=COLOR_AIS, lw=lw, ms=ms, label=tr("b11_ais"))

    ax.axvline(0.0, color=COLOR_GRID, lw=1.0, alpha=0.6)
    ax.set_xlabel(xlabel, color=COLOR_TEXT)
    ax.set_ylabel(tr("b11_ylabel_h"), color=COLOR_TEXT)
    ax.set_title(title, color=COLOR_TEXT, fontweight="bold")
    if mode_tag:
        ax.text(0.98, 0.02, str(mode_tag), transform=ax.transAxes,
                ha="right", va="bottom", color=COLOR_TEXT, fontsize=9, alpha=0.85)
    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.45)
    ax.tick_params(colors=COLOR_TEXT)
    ax.legend(facecolor=BG, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, loc="best")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    vmax = float(np.max(np.r_[x_fix, x_ais])) if len(x_fix) else 1.0
    vmax = 1.0 if (not np.isfinite(vmax) or vmax <= 0) else vmax
    ax.set_xlim(0.0, 1.10 * vmax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

def _pct_red(a, b):
    a = float(a); b = float(b)
    if (not np.isfinite(a)) or abs(a) < 1e-15 or (not np.isfinite(b)):
        return np.nan
    return (1.0 - (b / a)) * 100.0

def _fmt(x, nd=4):
    try:
        x = float(x)
        if not np.isfinite(x):
            return "—"
        return f"{x:.{nd}f}"
    except Exception:
        return "—"

# -------------------------------------------------------------------------
# Datos requeridos
# -------------------------------------------------------------------------
required_keys = [
    "cmp_V_fix_story", "cmp_V_ais_story", "cmp_tag_shear",
    "cmp_U_fix_levels", "cmp_U_ais_levels", "cmp_tag_disp",
    "cmp_drift_fix_levels", "cmp_drift_ais_levels", "cmp_tag_drift",
    "alturas",
]
missing = [k for k in required_keys if st.session_state.get(k, None) is None]
if missing:
    st.error(tr("b11_need").format(keys=", ".join(missing)))
    st.stop()

alt_fix = np.asarray(st.session_state.get("alturas"), float).ravel()
n_pisos = int(len(alt_fix))
y_levels = np.r_[0.0, alt_fix]

# Tags
tagV = str(st.session_state.get("cmp_tag_shear", "—"))
tagU = str(st.session_state.get("cmp_tag_disp", "—"))
tagD = str(st.session_state.get("cmp_tag_drift", "—"))

# -------------------------------------------------------------------------
# CORTANTES
# -------------------------------------------------------------------------
V_fix_env = np.asarray(st.session_state.get("cmp_V_fix_story"), float).ravel()
V_ais_env = np.asarray(st.session_state.get("cmp_V_ais_story"), float).ravel()
V_fix_max = np.asarray(st.session_state.get("cmp_V_fix_story_max", V_fix_env), float).ravel()
V_fix_min = st.session_state.get("cmp_V_fix_story_min", None)
V_ais_max = np.asarray(st.session_state.get("cmp_V_ais_story_max", V_ais_env), float).ravel()
V_ais_min = st.session_state.get("cmp_V_ais_story_min", None)

if V_fix_min is not None:
    V_fix_min = np.asarray(V_fix_min, float).ravel()
if V_ais_min is not None:
    V_ais_min = np.asarray(V_ais_min, float).ravel()

isV_maxmin = ("THA" in tagV and V_fix_min is not None and V_ais_min is not None)
modeV_tag = tagV

# -------------------------------------------------------------------------
# DESPLAZAMIENTOS
# -------------------------------------------------------------------------
U_fix_levels = np.asarray(st.session_state.get("cmp_U_fix_levels"), float).ravel()
U_ais_levels = np.asarray(st.session_state.get("cmp_U_ais_levels"), float).ravel()

U_fix_max = U_fix_levels.copy()
U_ais_max = U_ais_levels.copy()
U_fix_min = None
U_ais_min = None
isU_maxmin = False

if "THA" in tagU:
    U_fix_hist = st.session_state.get("cmp_U_fix_hist_levels", None)
    U_ais_hist = st.session_state.get("cmp_U_ais_hist_levels", None)

    if U_fix_hist is not None and U_ais_hist is not None:
        U_fix_hist = np.asarray(U_fix_hist, float)
        U_ais_hist = np.asarray(U_ais_hist, float)

        if U_fix_hist.ndim == 2 and U_fix_hist.shape[0] == len(y_levels):
            U_fix_max = np.max(U_fix_hist, axis=1)
            U_fix_min = np.min(U_fix_hist, axis=1)
        if U_ais_hist.ndim == 2 and U_ais_hist.shape[0] == len(y_levels):
            U_ais_max = np.max(U_ais_hist, axis=1)
            U_ais_min = np.min(U_ais_hist, axis=1)

        if U_fix_min is not None and U_ais_min is not None:
            isU_maxmin = True

modeU_tag = tagU

# -------------------------------------------------------------------------
# DERIVAS (desde Bloque 10 ya vienen como NEC24, con base = 0 y niveles)
# -------------------------------------------------------------------------
D_fix_levels = np.asarray(st.session_state.get("cmp_drift_fix_levels"), float).ravel()
D_ais_levels = np.asarray(st.session_state.get("cmp_drift_ais_levels"), float).ravel()
y_drift_fix  = np.asarray(st.session_state.get("cmp_drift_y_fix_levels", y_levels), float).ravel()
y_drift_ais  = np.asarray(st.session_state.get("cmp_drift_y_ais_levels", y_levels), float).ravel()

# para comparativo final usamos mismo eje vertical del modelo
if len(y_drift_fix) == len(y_levels):
    y_plot_d = y_drift_fix
else:
    y_plot_d = y_levels

# -------------------------------------------------------------------------
# Métricas resumen
# -------------------------------------------------------------------------
V0_fix = float(st.session_state.get("cmp_Vb_fix", np.nan))
V0_ais = float(st.session_state.get("cmp_Vb_ais", np.nan))

uH_fix = float(U_fix_max[-1]) if len(U_fix_max) else np.nan
uH_ais = float(U_ais_max[-1]) if len(U_ais_max) else np.nan

dmax_fix = float(np.max(np.asarray(D_fix_levels, float).ravel())) if len(D_fix_levels) else np.nan
dmax_ais = float(np.max(np.asarray(D_ais_levels, float).ravel())) if len(D_ais_levels) else np.nan

pfa_fix_g = np.asarray(st.session_state.get("pfa_fix_g", []), float).ravel()
pfa_ais_g = np.asarray(st.session_state.get("pfa_ais_g", []), float).ravel()
pfa_fix_max = float(np.max(np.abs(pfa_fix_g))) if len(pfa_fix_g) else np.nan
pfa_ais_max = float(np.max(np.abs(pfa_ais_g))) if len(pfa_ais_g) else np.nan

T_fix = np.asarray(st.session_state.get("T_sin", []), float).ravel()
T_ais = np.asarray(st.session_state.get("T_ais", []), float).ravel()
lambdaT = float(T_ais[0] / T_fix[0]) if (len(T_fix) and len(T_ais) and np.isfinite(T_fix[0]) and abs(T_fix[0]) > 1e-15 and np.isfinite(T_ais[0])) else np.nan

etaV = float(V0_ais / V0_fix) if (np.isfinite(V0_fix) and abs(V0_fix) > 1e-15 and np.isfinite(V0_ais)) else np.nan

u_iso_max = np.nan
u_cap = np.nan
iso_use = np.nan

# --------------------------------------------------------------
# Demanda del aislador:
# THA -> desde historia temporal
# RSA -> desde desplazamiento absoluto del aislador (nivel 0)
# --------------------------------------------------------------
u_ais_hist = st.session_state.get("cmp_U_ais_hist_levels", None)

if u_ais_hist is not None:
    u_ais_hist = np.asarray(u_ais_hist, float)
    if u_ais_hist.ndim == 2 and u_ais_hist.shape[0] >= 1:
        # THA: máximo absoluto en el tiempo del GDL del aislador
        u_iso_max = float(np.max(np.abs(u_ais_hist[0, :])))
else:
    # RSA: usar el desplazamiento absoluto del aislador en el perfil almacenado
    if len(U_ais_levels) >= 1 and np.isfinite(U_ais_levels[0]):
        u_iso_max = float(np.abs(U_ais_levels[0]))

for key in ("D_M", "u_cap", "u_capacidad", "u_cap_ais"):
    val = st.session_state.get(key, None)
    if val is not None:
        try:
            u_cap = float(val)
            break
        except Exception:
            pass

if np.isfinite(u_iso_max) and np.isfinite(u_cap) and abs(u_cap) > 1e-15:
    iso_use = float(u_iso_max / u_cap)

chg_V  = _pct_red(V0_fix, V0_ais)
chg_uH = _pct_red(abs(uH_fix), abs(uH_ais))
chg_d  = _pct_red(dmax_fix, dmax_ais)
chg_p  = _pct_red(pfa_fix_max, pfa_ais_max)

# -------------------------------------------------------------------------
# 📊 Layout 2x2
# -------------------------------------------------------------------------
colA, colB = st.columns([1, 1], gap="large")
colC, colD = st.columns([1, 1], gap="large")

# ========================= 1) CORTANTES =========================
with colA:
    with st.container(border=True):
        st.subheader(tr("b11_hdr_V"), help=tr("b11_help_v"))
        st.caption(tr("b11_tag_shear").format(tag=tagV))

        _plot_story_shear_compare(
            V_fix_max,
            V_fix_min if isV_maxmin else None,
            V_ais_max,
            V_ais_min if isV_maxmin else None,
            y_levels=y_levels,
            title=tr("b11_hdr_V"),
            n_pisos_ref=n_pisos,
            mode_tag=modeV_tag
        )

# ========================= 2) DESPLAZAMIENTOS =========================
with colB:
    with st.container(border=True):
        st.subheader(tr("b11_hdr_U"), help=tr("b11_help_u"))
        st.caption(tr("b11_tag_disp").format(tag=tagU))

        _plot_profile_compare(
            U_fix_max,
            U_fix_min if isU_maxmin else None,
            U_ais_max,
            U_ais_min if isU_maxmin else None,
            y_levels=y_levels,
            title=tr("b11_hdr_U"),
            n_pisos_ref=n_pisos,
            xlabel=tr("b11_xlabel_U"),
            mode_tag=modeU_tag
        )

# ========================= 3) DERIVAS =========================
with colC:
    with st.container(border=True):
        st.subheader(tr("b11_hdr_D"), help=tr("b11_help_d"))
        st.caption(tr("b11_tag_drift").format(tag=tagD))

        _plot_drift_compare_positive(
            D_fix_levels,
            D_ais_levels,
            y_levels=y_plot_d,
            title=tr("b11_hdr_D"),
            n_pisos_ref=n_pisos,
            xlabel=tr("b11_xlabel_D"),
            mode_tag=tagD
        )

# ========================= 4) RESUMEN FINAL =========================
with colD:
    with st.container(border=True):
        st.subheader(tr("b11_hdr_S"))

        lang_now = st.session_state.get("lang", "en")
        shear_reduction = chg_V if np.isfinite(chg_V) else np.nan

        if np.isfinite(iso_use):
            if iso_use <= 0.80:
                iso_status = "Acceptable" if lang_now == "en" else "Aceptable"
            elif iso_use <= 1.00:
                iso_status = "Near limit" if lang_now == "en" else "Cerca del límite"
            else:
                iso_status = "High demand" if lang_now == "en" else "Alta demanda"
        else:
            iso_status = "—"

        if np.isfinite(etaV) and np.isfinite(iso_use):
            if etaV <= 0.30 and iso_use <= 0.80:
                final_msg = "Effective and feasible isolation" if lang_now == "en" else "Aislamiento efectivo y viable"
            elif etaV <= 0.30 and iso_use > 0.80:
                final_msg = "Effective, but with high isolation demand" if lang_now == "en" else "Efectivo, pero con alta demanda en el aislamiento"
            elif etaV > 0.30:
                final_msg = "Limited benefit for this configuration" if lang_now == "en" else "Beneficio limitado para esta configuración"
            else:
                final_msg = "Check isolation performance" if lang_now == "en" else "Revisar desempeño del aislamiento"
        else:
            final_msg = "—"

        # --------------------------------------------------------------
        # Métricas principales
        # --------------------------------------------------------------
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Base shear reduction" if lang_now == "en" else "Reducción de cortante basal",
                f"{shear_reduction:.1f} %" if np.isfinite(shear_reduction) else "—"
            )
        with c2:
            st.metric(
                "Isolation demand" if lang_now == "en" else "Demanda del aislamiento",
                iso_status
            )

        st.markdown(f"**{'Interpretation' if lang_now == 'en' else 'Interpretación'}**")
        st.info(final_msg)

        # --------------------------------------------------------------
        # Tabla comparativa compacta y clara
        # --------------------------------------------------------------
        hdr_indicator = "Indicator" if lang_now == "en" else "Indicador"
        hdr_change = "Change" if lang_now == "en" else "Cambio"

        row_v_fix = _fmt(V0_fix, 4)
        row_v_ais = _fmt(V0_ais, 4)
        row_v_chg = f"{chg_V:.2f} %" if np.isfinite(chg_V) else "—"

        row_u_fix = _fmt(uH_fix, 4)
        row_u_ais = _fmt(uH_ais, 4)
        row_u_chg = f"{chg_uH:.2f} %" if np.isfinite(chg_uH) else "—"

        row_d_fix = _fmt(dmax_fix * 100.0, 3) + " %" if np.isfinite(dmax_fix) else "—"
        row_d_ais = _fmt(dmax_ais * 100.0, 3) + " %" if np.isfinite(dmax_ais) else "—"
        row_d_chg = f"{chg_d:.2f} %" if np.isfinite(chg_d) else "—"

        row_i_fix = "—"
        row_i_ais = _fmt(iso_use, 4)
        row_i_chg = "—"

        table_html = f"""
        <div style="
            border: 1px solid #D9D9D9;
            border-radius: 12px;
            overflow: hidden;
            margin-top: 8px;
            margin-bottom: 8px;
        ">
            <table style="
                width: 100%;
                border-collapse: collapse;
                font-size: 15px;
            ">
                <thead>
                    <tr style="background-color: #F7F7F9;">
                        <th style="text-align:left; padding:10px 12px; border-bottom:1px solid #E6E6E6;">{hdr_indicator}</th>
                        <th style="text-align:left; padding:10px 12px; border-bottom:1px solid #E6E6E6;">{tr("b11_fix")}</th>
                        <th style="text-align:left; padding:10px 12px; border-bottom:1px solid #E6E6E6;">{tr("b11_ais")}</th>
                        <th style="text-align:left; padding:10px 12px; border-bottom:1px solid #E6E6E6;">{hdr_change}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding:10px 12px; border-bottom:1px solid #EEEEEE;">{tr("b11_vbase")}</td>
                        <td style="padding:10px 12px; border-bottom:1px solid #EEEEEE;">{row_v_fix}</td>
                        <td style="padding:10px 12px; border-bottom:1px solid #EEEEEE;">{row_v_ais}</td>
                        <td style="padding:10px 12px; border-bottom:1px solid #EEEEEE;">{row_v_chg}</td>
                    </tr>
                    <tr>
                        <td style="padding:10px 12px; border-bottom:1px solid #EEEEEE;">{tr("b11_roof_u")}</td>
                        <td style="padding:10px 12px; border-bottom:1px solid #EEEEEE;">{row_u_fix}</td>
                        <td style="padding:10px 12px; border-bottom:1px solid #EEEEEE;">{row_u_ais}</td>
                        <td style="padding:10px 12px; border-bottom:1px solid #EEEEEE;">{row_u_chg}</td>
                    </tr>
                    <tr>
                        <td style="padding:10px 12px; border-bottom:1px solid #EEEEEE;">{tr("b11_drift_max")}</td>
                        <td style="padding:10px 12px; border-bottom:1px solid #EEEEEE;">{row_d_fix}</td>
                        <td style="padding:10px 12px; border-bottom:1px solid #EEEEEE;">{row_d_ais}</td>
                        <td style="padding:10px 12px; border-bottom:1px solid #EEEEEE;">{row_d_chg}</td>
                    </tr>
                    <tr>
                        <td style="padding:10px 12px;">{tr("b11_iso_use")}</td>
                        <td style="padding:10px 12px;">{row_i_fix}</td>
                        <td style="padding:10px 12px;">{row_i_ais}</td>
                        <td style="padding:10px 12px;">{row_i_chg}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """

        st.markdown(table_html, unsafe_allow_html=True)

        # Espacio final para empatar mejor con la altura del gráfico izquierdo
        st.markdown("<div style='height: 110px;'></div>", unsafe_allow_html=True)

# st.success(tr("b11_ok"))
