import numpy as np
from typing import Union, Optional, Tuple, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib import rcParams
import re
import math
from math import atan2, degrees
from scipy import signal

# ==========================================================
# === MATRIZ DE RIGIDEZ LOCAL 6x6 – ELEMENTO VIGA-COLUMNA 2D
# ==========================================================
def beam_stiffness_2D(E: float, I: float, A: float, L: float) -> np.ndarray:
    """
    Calcula la matriz de rigidez local 6x6 para un elemento viga-columna 2D.
    Teoría de Euler–Bernoulli. Orden de GDL: [u1, v1, θ1, u2, v2, θ2]
    """
    if L <= 0:
        raise ValueError("La longitud del elemento debe ser mayor a cero.")
    return np.array([
        [ E*A/L,      0,            0,     -E*A/L,       0,            0],
        [ 0,      12*E*I/L**3,   6*E*I/L**2,   0,   -12*E*I/L**3,   6*E*I/L**2],
        [ 0,       6*E*I/L**2,   4*E*I/L,      0,    -6*E*I/L**2,   2*E*I/L],
        [-E*A/L,      0,            0,      E*A/L,       0,            0],
        [ 0,     -12*E*I/L**3,  -6*E*I/L**2,  0,    12*E*I/L**3,  -6*E*I/L**2],
        [ 0,       6*E*I/L**2,   2*E*I/L,     0,    -6*E*I/L**2,   4*E*I/L]
    ], dtype=float)


# ==========================================================
# === MATRIZ DE TRANSFORMACIÓN 6x6 – LOCAL ↔ GLOBAL
# ==========================================================
def transformation_matrix(node_start: tuple, node_end: tuple) -> np.ndarray:
    """
    Calcula la matriz de transformación 6x6 para un elemento viga-columna 2D.
    node_start, node_end: (x, y)
    """
    dx = node_end[0] - node_start[0]
    dy = node_end[1] - node_start[1]
    L = np.hypot(dx, dy)
    if L == 0:
        raise ValueError("Los nodos del elemento no pueden coincidir (L = 0).")
    c, s = dx / L, dy / L
    return np.array([
        [ c,  s, 0,   0,  0, 0],
        [-s,  c, 0,   0,  0, 0],
        [ 0,  0, 1,   0,  0, 0],
        [ 0,  0, 0,   c,  s, 0],
        [ 0,  0, 0,  -s,  c, 0],
        [ 0,  0, 0,   0,  0, 1]
    ], dtype=float)

# ==========================================================
# === CLASE ELEMENTO VIGA-COLUMNA 2D
# ==========================================================
class Element:
    """
    Representa un elemento viga-columna 2D en un modelo estructural.
    Contiene propiedades, matrices local/global y mapeo de GDL.
    """

    def __init__(self, node_start, node_end, E, I, A, gdl_map):
        self.node_start = node_start
        self.node_end = node_end
        self.E, self.I, self.A = E, I, A
        self.length = np.hypot(node_end[0] - node_start[0],
                               node_end[1] - node_start[1])
        if self.length == 0:
            raise ValueError("Elemento con longitud nula detectado.")

        self.k_local = beam_stiffness_2D(E, I, A, self.length)
        self.T = transformation_matrix(node_start, node_end)
        self.k_global = self.T.T @ self.k_local @ self.T
        self.dofs = self.assign_dofs(gdl_map)

    def assign_dofs(self, gdl_map: dict) -> list:
        """
        Devuelve [vx_i, vy_i, θ_i, vx_j, vy_j, θ_j] según gdl_map.
        """
        i, j = self.node_start[2], self.node_end[2]
        dofs_i = [gdl_map.get((i, d)) for d in ('vx', 'vy', 'theta')]
        dofs_j = [gdl_map.get((j, d)) for d in ('vx', 'vy', 'theta')]
        return dofs_i + dofs_j

# ==========================================================
# === ENSAMBLE DE MATRIZ DE RIGIDEZ GLOBAL (FULL 3GDL/NODO)
# ==========================================================
def assemble_global_stiffness(elements: list, total_dofs: int) -> np.ndarray:
    """
    Ensambla la matriz global considerando 3GDL por nodo: [vx, vy, theta].
    El 'total_dofs' debe ser el máximo índice+1 de gdl_map.
    """
    K = np.zeros((total_dofs, total_dofs), dtype=float)

    # Local/global de elemento 2D Euler-Bernoulli:
    # [u1, v1, θ1, u2, v2, θ2]  -> aquí u=vx, v=vy
    for el in elements:
        ke = el.k_global
        dofs = el.dofs  # [vx_i, vy_i, th_i, vx_j, vy_j, th_j]

        for a in range(6):
            da = dofs[a]
            if da is None:
                continue
            for b in range(6):
                db = dofs[b]
                if db is None:
                    continue
                K[int(da), int(db)] += ke[a, b]

    return K

# ==========================================================
# === GENERAR MAPA DE GRADOS DE LIBERTAD (GDL)
# ==========================================================
def generar_gdl_map(nodes: list) -> dict:
    """
    Genera { (id_nodo, tipo) : Optional[\1] }.

    ✅ Modelo recomendado (ETABS-like en 2D):
    - vx: 1 GDL por piso (diafragma rígido en X)
    - vy: 1 GDL por nodo
    - theta: 1 GDL por nodo
    Base (y=0): empotrada -> None
    """
    # pisos (sin base)
    niveles = sorted({round(float(y), 6) for (_, y, _) in nodes if float(y) > 0.0})
    n_pisos = len(niveles)

    # vx por piso (0..n_pisos-1)
    vx_por_nivel = {nivel: i for i, nivel in enumerate(niveles)}
    counter = n_pisos  # desde aquí arrancan vy/theta por nodo

    gdl_map = {}
    for (x, y, nid) in nodes:
        y = round(float(y), 6)
        if y == 0.0:
            gdl_map[(nid, 'vx')] = None
            gdl_map[(nid, 'vy')] = None
            gdl_map[(nid, 'theta')] = None
        else:
            # ✅ vx compartido por piso
            gdl_map[(nid, 'vx')] = vx_por_nivel[y]

            # ✅ vy y theta por nodo
            gdl_map[(nid, 'vy')] = counter; counter += 1
            gdl_map[(nid, 'theta')] = counter; counter += 1

    return gdl_map

# ==========================================================
# === INTEGRADOR DINÁMICO NEWMARK-BETA (IMPLÍCITO) =========
# ==========================================================
def newmark(M: np.ndarray, C: np.ndarray, K: np.ndarray,
            U0: np.ndarray, V0: np.ndarray, dt: float, Pt: np.ndarray,
            gamma: float = 0.5, beta: float = 0.25):
    """
    Integra M·Ü + C·Û + K·U = P(t) con Newmark-Beta (implícito, aceleración constante).
    Retorna U, V, A con dimensiones (nDOF, nSteps).
    Requisitos de estabilidad incondicional: gamma >= 0.5 y beta >= 0.25.
    """
    if dt <= 0:
        raise ValueError("dt debe ser > 0.")
    if gamma < 0.5 or beta < 0.25:
        raise ValueError("Para estabilidad incondicional use gamma>=0.5 y beta>=0.25.")

    M = np.asarray(M, float)
    C = np.asarray(C, float)
    K = np.asarray(K, float)
    U0 = np.asarray(U0, float).ravel()
    V0 = np.asarray(V0, float).ravel()
    Pt = np.asarray(Pt, float)

    n = K.shape[0]
    if Pt.ndim == 1:
        Pt = Pt.reshape(n, -1)
    if Pt.shape[0] != n and Pt.shape[1] == n:
        Pt = Pt.T
    if Pt.shape[0] != n:
        raise ValueError("Pt debe tener forma (nDOF, nSteps).")

    nSteps = Pt.shape[1]
    U = np.zeros((n, nSteps), float)
    V = np.zeros((n, nSteps), float)
    A = np.zeros((n, nSteps), float)

    # Aceleración inicial por equilibrio dinámico
    try:
        A0 = np.linalg.solve(M, Pt[:, 0] - C @ V0 - K @ U0)
    except np.linalg.LinAlgError:
        A0 = np.zeros_like(U0)
    A0 = np.nan_to_num(A0, nan=0.0, posinf=0.0, neginf=0.0)

    U[:, 0] = U0
    V[:, 0] = V0
    A[:, 0] = A0

    # Coeficientes Newmark
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    # Rigidez efectiva (constante en el tiempo para sistema lineal)
    Khat = K + a0 * M + a1 * C

    # Factorización (Cholesky si se puede; si no, solve paso a paso)
    use_chol = False
    L = None
    try:
        L = np.linalg.cholesky(Khat)
        use_chol = True
    except np.linalg.LinAlgError:
        pass

    for i in range(nSteps - 1):
        # Fuerza efectiva
        Peff = (Pt[:, i + 1]
                + M @ (a0 * U[:, i] + a2 * V[:, i] + a3 * A[:, i])
                + C @ (a1 * U[:, i] + a4 * V[:, i] + a5 * A[:, i]))

        # Resolver U_{i+1}
        if use_chol:
            # Resolver Khat * x = Peff vía Cholesky
            y = np.linalg.solve(L, Peff)
            U[:, i + 1] = np.linalg.solve(L.T, y)
        else:
            try:
                U[:, i + 1] = np.linalg.solve(Khat, Peff)
            except np.linalg.LinAlgError:
                # Fallback robusto
                U[:, i + 1] = np.linalg.pinv(Khat) @ Peff

        # Aceleración y velocidad en i+1 (formulación estándar)
        A[:, i + 1] = a0 * (U[:, i + 1] - U[:, i]) - a2 * V[:, i] - a3 * A[:, i]
        V[:, i + 1] = V[:, i] + dt * ((1.0 - gamma) * A[:, i] + gamma * A[:, i + 1])

    return U, V, A

# =============================================================================
# === GRAFICADO DE RESPUESTAS POR PISO (ARCTIC DARK – STREAMLIT/JUPYTER) =====
# =============================================================================
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

def graficar_respuesta_por_piso(
    t: np.ndarray,
    u_t: np.ndarray,
    v_t: np.ndarray,
    a_t: np.ndarray,
    alturas: Union[np.ndarray, list],
    t_total: float,
    *args, **kwargs
):
    """
    Grafica para cada piso: desplazamiento, velocidad y aceleración.

    ✅ AUTO-BILINGÜE sin depender de tr():
    - Lee st.session_state.lang si existe ("en" o "es").
    - Si NO existe streamlit/session_state, usa español por defecto.
    - Puedes forzar textos pasando kwargs (opcional).

    Kwargs opcionales:
      - nombre_piso: str
      - ylabel_u, ylabel_v, ylabel_a: str
      - xlabel: str
      - title_fmt: str  (usa {piso})
      - max_label / min_label: str  (prefijo para leyenda)
    """
    nombre_piso = kwargs.get("nombre_piso", None)

    # ---------------------------
    # Streamlit + idioma
    # ---------------------------
    _HAS_ST = False
    _lang = "es"  # default
    try:
        import streamlit as st  # type: ignore
        _HAS_ST = True
        _lang = st.session_state.get("lang", "en")  # tu app: "en" por defecto
    except Exception:
        st = None  # type: ignore

    if _lang not in ("en", "es"):
        _lang = "en"

    # ---------------------------
    # Defaults por idioma (si NO pasas kwargs)
    # ---------------------------
    if _lang == "en":
        _d_ylabel_u = "Displacement [m]"
        _d_ylabel_v = "Velocity [m/s]"
        _d_ylabel_a = "Acceleration [m/s²]"
        _d_xlabel   = "Time [s]"
        _d_titlefmt = "Floor responses – {piso}"
        _d_max = "Max"
        _d_min = "Min"
    else:
        _d_ylabel_u = "Desplazamiento [m]"
        _d_ylabel_v = "Velocidad [m/s]"
        _d_ylabel_a = "Aceleración [m/s²]"
        _d_xlabel   = "Tiempo [s]"
        _d_titlefmt = "Respuestas del piso {piso}"
        _d_max = "Máx"
        _d_min = "Mín"

    ylabel_u = kwargs.get("ylabel_u", _d_ylabel_u)
    ylabel_v = kwargs.get("ylabel_v", _d_ylabel_v)
    ylabel_a = kwargs.get("ylabel_a", _d_ylabel_a)
    xlabel   = kwargs.get("xlabel", _d_xlabel)
    title_fmt = kwargs.get("title_fmt", _d_titlefmt)
    max_label = kwargs.get("max_label", _d_max)
    min_label = kwargs.get("min_label", _d_min)

    etiquetas = [ylabel_u, ylabel_v, ylabel_a]

    # ---------------------------
    # 🎨 Paleta Arctic Dark Pastel
    # ---------------------------
    COLOR_BG    = '#2B3141'
    COLOR_TEXT  = '#E8EDF2'
    COLOR_GRID  = '#5B657A'
    COLOR_LINE  = ['#7EB6FF', '#FFD180', '#A8E6CF']
    COLOR_MAX   = '#FFB3B3'
    COLOR_MIN   = '#89D6FF'
    LEG_FACE    = '#363C4A'
    LEG_EDGE    = '#A7B1C5'

    # ---------------------------
    # Datos
    # ---------------------------
    t = np.asarray(t, dtype=float).ravel()
    u_t = np.asarray(u_t, dtype=float)
    v_t = np.asarray(v_t, dtype=float)
    a_t = np.asarray(a_t, dtype=float)
    n_pisos = u_t.shape[0]

    respuestas = [u_t, v_t, a_t]

    for i in range(n_pisos):
        fig, axs = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
        fig.patch.set_facecolor(COLOR_BG)

        for j in range(3):
            serie = np.asarray(respuestas[j][i], dtype=float).ravel()
            ax = axs[j]
            ax.set_facecolor(COLOR_BG)

            ax.plot(t, serie, color=COLOR_LINE[j], linewidth=1.0, alpha=0.95)

            # --- Máximo y mínimo ---
            max_idx = int(np.argmax(serie))
            min_idx = int(np.argmin(serie))
            max_t, max_val = float(t[max_idx]), float(serie[max_idx])
            min_t, min_val = float(t[min_idx]), float(serie[min_idx])

            ax.plot(max_t, max_val, 'o', color=COLOR_MAX, markersize=4.5)
            ax.plot(min_t, min_val, 's', color=COLOR_MIN, markersize=4.5)

            legend_elements = [
                Line2D([0], [0], marker='o', color='w',
                       label=f'{max_label}: {max_val:.4e} @ {max_t:.2f}s',
                       markerfacecolor=COLOR_MAX, markersize=6),
                Line2D([0], [0], marker='s', color='w',
                       label=f'{min_label}: {min_val:.4e} @ {min_t:.2f}s',
                       markerfacecolor=COLOR_MIN, markersize=6)
            ]
            leg = ax.legend(handles=legend_elements, loc="upper right",
                            facecolor=LEG_FACE, edgecolor=LEG_EDGE,
                            framealpha=0.9, fontsize=8.5)
            for tt in leg.get_texts():
                tt.set_color(COLOR_TEXT)

            ax.set_ylabel(etiquetas[j], color=COLOR_TEXT, fontsize=9)
            ax.grid(True, color=COLOR_GRID, linestyle=':', alpha=0.35)
            ax.tick_params(colors=COLOR_TEXT, labelsize=8)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color(COLOR_GRID)
            ax.spines['left'].set_color(COLOR_GRID)

        axs[-1].set_xlabel(xlabel, color=COLOR_TEXT, fontsize=9)
        axs[-1].set_xlim(0, float(t_total))

        piso_txt = str(nombre_piso) if nombre_piso is not None else str(i + 1)
        fig.suptitle(title_fmt.format(piso=piso_txt),
                     fontsize=12, color=COLOR_TEXT, fontweight='semibold')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if _HAS_ST and st is not None:
            st.pyplot(fig)
        else:
            plt.show()

        plt.close(fig)

# ==========================================================
# === LECTOR DE MATRIZ NUMÉRICA CON RELLENO AUTOMÁTICO =====
# ==========================================================
def generar_matriz_con_relleno(contenido: str, filas_a_saltar: int) -> np.ndarray:
    """
    Convierte texto en matriz NumPy, rellenando filas cortas con ceros.
    """
    lineas = contenido.splitlines()
    datos = lineas[filas_a_saltar:]

    matriz = []
    max_columnas = 0
    for linea in datos:
        s = linea.strip()
        if not s:
            continue
        # Reemplaza separadores comunes por espacio
        s = s.replace(',', ' ')
        try:
            numeros = [float(x) for x in s.split()]
        except ValueError:
            continue
        if numeros:
            matriz.append(numeros)
            max_columnas = max(max_columnas, len(numeros))

    matriz_completa = [fila + [0.0] * (max_columnas - len(fila)) for fila in matriz]
    return np.array(matriz_completa, dtype=float)


# ===============================================================
# === REORGANIZAR MATRIZ EN SERIE TEMPORAL CONTINUA + TIEMPO ====
# ===============================================================
def reorganizar_con_tiempo(matriz: np.ndarray, num_columnas_usar: int, dt: float) -> pd.DataFrame:
    """
    Une columnas por filas en una sola serie y genera vector de tiempo asociado.
    """
    if dt <= 0:
        raise ValueError("dt debe ser > 0.")
    matriz = np.asarray(matriz, dtype=float)
    if matriz.ndim != 2:
        raise ValueError("matriz debe ser 2D.")
    num_columnas_usar = int(min(num_columnas_usar, matriz.shape[1]))
    submatriz = matriz[:, :num_columnas_usar]
    datos = submatriz.flatten(order='C')
    tiempo = np.arange(len(datos), dtype=float) * dt
    return pd.DataFrame({"Tiempo (s)": tiempo, "Aceleración": datos})


# =============================================================
# === ANÁLISIS MODAL: PERÍODOS, FRECUENCIAS Y MODOS NORMALES ==
# =============================================================
def orden_eig(K: np.ndarray, M: np.ndarray, normalizar_masa: bool = False):
    """
    Resuelve K·Φ = M·Φ·ω² y retorna T [s], modos normalizados y f [Hz].
    """
    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)

    # Problema generalizado (forma clásica). Para mayor robustez se podría usar scipy.linalg.eigh(K, M).
    w2, V = np.linalg.eig(np.linalg.inv(M) @ K)

    # Asegurar reales y no negativos por efectos numéricos
    w2 = np.real(w2)
    w2 = np.where(w2 < 0, np.abs(w2), w2)

    omega = np.sqrt(w2 + 0.0)
    idx = np.argsort(omega)
    omega = omega[idx]
    V = np.real(V[:, idx])

    # Evitar división por cero si hay modo rígido (omega≈0)
    omega = np.where(omega <= 1e-12, 1e-12, omega)
    frecuencias = omega / (2 * np.pi)
    T = 2 * np.pi / omega

    if normalizar_masa:
        for i in range(V.shape[1]):
            m_gen = float(V[:, i].T @ M @ V[:, i])
            if m_gen <= 0:
                continue
            V[:, i] = V[:, i] / np.sqrt(m_gen)

    modos_norm = np.zeros_like(V)
    for i in range(V.shape[1]):
        base = V[0, i]
        esc = 1.0 if abs(base) < 1e-15 else np.abs(base)
        modo = V[:, i] / esc
        if modo[0] < 0:
            modo = -modo
        modos_norm[:, i] = modo

    return T, modos_norm, frecuencias

# =======================================================================
# === MATRIZ DE MASAS CONDENSADA POR PISO (MODELO 2D, LUMPED MASS) =====
# =======================================================================
def calcular_matriz_masas_por_piso(nodes: list,
                                   element_node_pairs: list,
                                   propiedades: dict,
                                   peso_especifico: float = 2.4028,
                                   sobrecarga_muerta: float = 0.0,
                                   b_col_x: float = 0.50) -> np.ndarray:
    """
    Masa lumped por piso (1 GDL/piso).

    Criterio:
    - Columnas: masa repartida 50% y 50% entre extremos, con longitud eje a eje.
    - Vigas: peso propio usando longitud libre sin solape con columnas.
    - Carga adicional lineal: usando longitud eje a eje completa.

    Unidades:
      peso_especifico [Tf/m³], A [m²], L [m] => peso [Tf], masa = peso/g [Tf·s²/m]
    """
    g = 9.8066500000

    alturas = sorted({round(y, 6) for (x, y, _) in nodes if y > 0})
    n_pisos = len(alturas)
    masas_por_piso = np.zeros(n_pisos, dtype=float)

    node_by_id = {nid: (x, y, nid) for (x, y, nid) in nodes}
    y_to_idx = {y: i for i, y in enumerate(alturas)}

    for n1, n2, tipo in element_node_pairs:
        x1, y1, _ = node_by_id[n1]
        x2, y2, _ = node_by_id[n2]

        A = float(propiedades[tipo]["A"])

        if tipo == "col":
            L = float(np.hypot(x2 - x1, y2 - y1))
            peso_elem = peso_especifico * A * L

            altura_inf = round(min(y1, y2), 6)
            altura_sup = round(max(y1, y2), 6)

            if altura_inf > 0:
                masas_por_piso[y_to_idx[altura_inf]] += (peso_elem / 2.0) / g
            if altura_sup > 0:
                masas_por_piso[y_to_idx[altura_sup]] += (peso_elem / 2.0) / g

        elif tipo == "viga":
            L = abs(float(x2 - x1))  # longitud eje a eje
            L_real = max(L - 0.5 * b_col_x - 0.5 * b_col_x, 0.0)  # longitud libre

            # peso propio de viga
            peso_elem = peso_especifico * A * L_real

            altura_viga = round(y1, 6)
            if altura_viga in y_to_idx:
                idx = y_to_idx[altura_viga]
                masas_por_piso[idx] += (peso_elem / g)

                # carga adicional lineal
                if sobrecarga_muerta > 0:
                    peso_sob = sobrecarga_muerta * L
                    masas_por_piso[idx] += (peso_sob / g)

    return np.diag(masas_por_piso)

def plot_structure(nodes, elements, nodos_restringidos,
                   gdl_dinamicos_local=None, gdl_estaticos_local=None,
                   gdl_map=None, propiedades=None,
                   title=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import matplotlib.transforms as mtrans
    from matplotlib.patches import Polygon, Rectangle

    # ====== TU PALETA ======
    BG_FIG   = "#2B3141"
    BG_AX    = "#2B3141"
    GRID     = "#5B657A"
    TEXT     = "#E8EDF2"
    ELEM_C   = "#A8D5FF"
    NODE_TXT = "#FFE6A3"
    HALO = [pe.withStroke(linewidth=2.0, foreground=BG_AX)]

    if not nodes or not elements:
        return None

    # ---- helpers robustos (por si llegan np.array) ----
    def _to_float(x): return float(np.asarray(x).reshape(-1)[0])
    def _to_int(x):   return int(np.asarray(x).reshape(-1)[0])

    xs = np.array([_to_float(n[0]) for n in nodes], dtype=float)
    ys = np.array([_to_float(n[1]) for n in nodes], dtype=float)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    Lx = max(x_max - x_min, 1e-9)
    Ly = max(y_max - y_min, 1e-9)

    # niveles para saber #pisos (sin asumir h constante)
    y_levels = sorted({round(_to_float(y), 8) for (x, y, nid) in nodes})
    n_pisos = max(len([yy for yy in y_levels if yy > y_min + 1e-9]), 1)

    # n_vanos por niveles (aprox): #x únicos - 1
    x_levels = sorted({round(_to_float(x), 8) for (x, y, nid) in nodes})
    n_vanos = max(len(x_levels) - 1, 1)

    # labels desde 1 si internamente empiezas en 0
    min_nid = min(_to_int(n[2]) for n in nodes)
    label_add = 1 if min_nid == 0 else 0

    # ====== FIGURA: cuadrada tipo “opsvis crudo bonito” ======
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=180)
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_AX)

    # ==========================
    # 1) DIBUJO DE ELEMENTOS (sin OpenSees)
    # ==========================
    # Dibujamos líneas directamente desde tus objetos `elements`
    # (espera el.node_start = (x,y,...) y el.node_end = (x,y,...))
    for el in elements:
        x1 = _to_float(el.node_start[0]); y1 = _to_float(el.node_start[1])
        x2 = _to_float(el.node_end[0]);   y2 = _to_float(el.node_end[1])
        # evitar elementos degenerados
        if (abs(x2 - x1) < 1e-12) and (abs(y2 - y1) < 1e-12):
            continue
        ax.plot([x1, x2], [y1, y2], color=ELEM_C, linewidth=1.0, solid_capstyle="round", zorder=2)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, alpha=0.35)

    # ==========================
    # 2) NODOS (scatter) + “soportes” como patches
    # ==========================
    # Lógica de escala igual a la tuya
    scale = float(np.clip(1.00 - 0.023 * n_pisos, 0.30, 1.00))

    # --- 1) elementos: ajustar lw en TODAS las líneas (manteniendo tu lógica)
    lw_elem = float(np.clip((1.35 - 0.02 * n_pisos) * scale, 0.55, 1.25))
    for ln in ax.lines:
        ln.set_color(ELEM_C)
        ln.set_linewidth(lw_elem)

    # --- 2) nodos (scatter collections)
    base_node = float(np.clip(14.0 - 0.38 * n_pisos, 2.6, 10.0))
    node_size = base_node * (scale**2)
    sup_size  = node_size * 1.05

    # scatter único de nodos (así controlas tamaños y estilos)
    xN = np.array([_to_float(n[0]) for n in nodes], dtype=float)
    yN = np.array([_to_float(n[1]) for n in nodes], dtype=float)

    is_base_all = np.isclose(yN, y_min, atol=1e-6)
    sizes = np.where(is_base_all, sup_size, node_size).astype(float)

    sc = ax.scatter(
        xN, yN,
        s=sizes,
        facecolor=ELEM_C,
        edgecolor=TEXT,
        linewidth=float(np.clip(0.45 * scale, 0.18, 0.45)),
        alpha=1.0,
        zorder=5
    )

    # --- 3) empotramientos/soportes (patches) en nodos restringidos
    # Creamos patches para que tu bloque de patches exista y mantenga “la lógica”
    if nodos_restringidos:
        # mapa nid -> (x,y)
        nid_to_xy = { _to_int(nid): (_to_float(x), _to_float(y)) for (x,y,nid) in nodes }

        # tamaño del símbolo relativo a la geometría
        # (similar a un “base band”)
        sym = 0.035 * max(Lx, Ly, 1.0)  # tamaño base
        tri_h = 0.55 * sym
        tri_w = 0.65 * sym
        base_h = 0.18 * sym

        for nid in nodos_restringidos:
            nid = _to_int(nid)
            if nid not in nid_to_xy:
                continue
            x0, y0 = nid_to_xy[nid]

            # solo dibujamos soporte si está en la base (como antes)
            if not np.isclose(y0, y_min, atol=1e-6):
                continue

            # rectángulo base (tipo “zapata”)
            rect = Rectangle(
                (x0 - 0.55*tri_w, y0 - base_h),
                width=1.10*tri_w,
                height=base_h,
                facecolor=ELEM_C,
                edgecolor=TEXT,
                linewidth=float(np.clip(0.55 * scale, 0.20, 0.55)),
                zorder=4
            )
            ax.add_patch(rect)

            # triangulito de empotramiento
            tri = Polygon(
                [[x0 - 0.5*tri_w, y0],
                 [x0 + 0.5*tri_w, y0],
                 [x0,            y0 - tri_h]],
                closed=True,
                facecolor=ELEM_C,
                edgecolor=TEXT,
                linewidth=float(np.clip(0.55 * scale, 0.20, 0.55)),
                zorder=4
            )
            ax.add_patch(tri)

    # --- 4) empotramientos/soportes (patches): escalar como tu código
    base_band = 0.10 * max(Ly, 1.0)
    for p in ax.patches:
        try:
            bb = p.get_extents()
            cy = 0.5 * (bb.y0 + bb.y1)
            if abs(cy - y_min) <= base_band:
                cx = 0.5 * (bb.x0 + bb.x1)
                tr = (mtrans.Affine2D()
                      .translate(-cx, -cy)
                      .scale(scale, scale)
                      .translate(cx, cy))
                p.set_transform(tr + p.get_transform())

            p.set_facecolor(ELEM_C)
            p.set_edgecolor(TEXT)
            p.set_linewidth(float(np.clip(0.55 * scale, 0.20, 0.55)))
            p.set_alpha(1.0)
        except Exception:
            pass

    # ==========================
    # NUMERACIÓN ULTRA COMPACTA (30 pisos) — igual que tú
    # ==========================
    if n_pisos <= 10:
        font_node = 8; step_i = 1; weight = "bold"
    elif n_pisos <= 18:
        font_node = 7; step_i = 2; weight = "bold"
    elif n_pisos <= 26:
        font_node = 6; step_i = 3; weight = "bold"
    else:
        font_node = 4; step_i = 5; weight = "normal"

    dx = 0.020 * max(Lx, 1.0)
    dy = float(np.clip(0.018 * (Ly / n_pisos), 0.08, 0.22))

    y_levels_r = [round(v, 8) for v in y_levels]
    last_level = len(y_levels_r) - 1

    for (x, y, nid) in nodes:
        x = _to_float(x); y = _to_float(y); nid = _to_int(nid)
        yl = round(y, 8)

        try:
            i_level = y_levels_r.index(yl)
        except ValueError:
            i_level = 0

        if (i_level % step_i) != 0 and i_level != last_level:
            continue

        ax.text(
            x + dx, y + dy, str(nid + label_add),
            fontsize=font_node, color=NODE_TXT, fontweight=weight,
            ha="left", va="center", path_effects=HALO, zorder=50
        )

    # ==========================
    # ✅ LÍMITES CUADRADOS — igual que tú
    # ==========================
    span = max(Lx, Ly)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    margin = 0.12 * span
    ax.set_xlim(cx - 0.5 * span - margin, cx + 0.5 * span + margin)
    ax.set_ylim(cy - 0.5 * span - margin, cy + 0.5 * span + margin)

    # ==========================
    # ✅ TÍTULO (si lo pasas)
    # ==========================
    if title is None:
        title = f"Pórtico 2D – {n_pisos} pisos, {n_vanos} vanos"
    ax.set_title(title, fontsize=11, color=TEXT, pad=6)

    ax.set_xlabel("X [m]", color=TEXT, fontsize=10)
    ax.set_ylabel("Y [m]", color=TEXT, fontsize=10)
    ax.tick_params(axis="both", labelsize=9, colors=TEXT)

    fig.tight_layout()
    return fig

# ============================================================
# === CÁLCULO DE VALORES MÁXIMOS ABSOLUTOS POR PISO O GDL ====
# ============================================================
def max_abs_por_piso(u_t, v_t, a_t):
    """Devuelve máximos absolutos de u, v, a por piso."""
    return (np.max(np.abs(u_t), axis=1),
            np.max(np.abs(v_t), axis=1),
            np.max(np.abs(a_t), axis=1))


# ============================================================
# === CONVERSIÓN A ACELERACIÓN ABSOLUTA ======================
# ============================================================
def abs_accel(a_rel, ag_g):
    """Convierte aceleración relativa [m/s²] a absoluta sumando la del terreno."""
    return a_rel + (ag_g * 9.8066500000)[np.newaxis, :]


# ============================================================
# === CORTANTE POR PISO EN EL TIEMPO =========================
# ============================================================
def story_shear_time_history(M, a_abs):
    """Calcula el cortante dinámico por piso V_i(t) = Σ_{j≥i} m_j·a_j(t)."""
    n, nt = a_abs.shape
    m = np.diag(M).reshape(-1, 1)
    F_iner = m * a_abs
    V = np.zeros_like(F_iner)
    V[-1, :] = F_iner[-1, :]
    for i in range(n - 2, -1, -1):
        V[i, :] = V[i + 1, :] + F_iner[i, :]
    return V

# ==============================================================
# === FORMATEO DE DATOS PARA GRÁFICO ESCALONADO POR PISOS =====
# ==============================================================
def escalera_xy(Vmax, alturas_levels):
    """
    Prepara los datos de cortante máximo por piso (o cualquier magnitud
    distribuida verticalmente) para graficar un diagrama escalonado.
    """
    Vmax = np.array(Vmax, dtype=float)
    y = np.array(alturas_levels, dtype=float)

    # Igualar longitudes
    n = min(len(Vmax), len(y))
    Vmax, y = Vmax[:n], y[:n]

    # Coordenadas extendidas (forma escalonada)
    x = np.concatenate([[Vmax[0]], Vmax])
    y = np.concatenate([[0.0], y])

    if len(x) > len(y):
        x = x[:len(y)]

    return x, y


# =======================================================================
# === MODELO BILINEAL HISTÉRETICO CON MEMORIA Y RIGIDEZ TANGENTE =======
# =======================================================================
def _bilinear_state(u0, u0_prev, uy, k0, kp, ue_prev):
    """
    Evalúa el estado interno de un elemento bilineal con memoria histerética.
    """
    du = u0 - u0_prev
    ue_trial = ue_prev + du
    ue = np.clip(ue_trial, -uy, uy)
    d_ue_du = 0.0 if (ue != ue_trial) else 1.0
    F_hyst = kp * u0 + (k0 - kp) * ue
    k_t = kp + (k0 - kp) * d_ue_du
    return F_hyst, k_t, ue


# =====================================================================================
# === NEWMARK NO LINEAL (AISLADOR BILINEAL EN LA BASE, DOF 0 DEL SISTEMA) ============
# =====================================================================================
def newmark_nl_base_bilinear(M, C, K, dt, ag_g,
                             k0, kp, Fy, c_iso,
                             u0_init=0.0, v0_init=0.0,
                             gamma=0.5, beta=0.25,
                             newton_tol=1e-6, newton_maxit=25):
    """
    Integra en el tiempo un sistema MDOF con aislador bilineal en el DOF 0
    (Newmark-Beta + Newton-Raphson).
    """
    n = M.shape[0]
    ag = ag_g * 9.8066500000
    nt = len(ag)

    # Inicialización
    U = np.zeros((n, nt))
    V = np.zeros((n, nt))
    A = np.zeros((n, nt))
    U[0, 0] = u0_init
    V[0, 0] = v0_init

    P0 = - (M @ np.ones((n, 1))).flatten() * ag[0]
    uy = Fy / k0
    ue_prev = 0.0
    F_h0, k_t0, ue_prev = _bilinear_state(U[0, 0], U[0, 0], uy, k0, kp, ue_prev)
    F_iso0 = F_h0 + c_iso * V[0, 0]

    A[:, 0] = np.linalg.solve(
        M,
        P0 - C @ V[:, 0] - K @ U[:, 0] - F_iso0 * np.r_[1.0, np.zeros(n - 1)]
    )

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2 * beta) - 1.0)

    e0 = np.zeros(n)
    e0[0] = 1.0

    F_iso_hist = np.zeros(nt)
    F_hyst_hist = np.zeros(nt)
    E_hyst = np.zeros(nt)
    F_iso_hist[0] = F_iso0
    F_hyst_hist[0] = F_h0

    M_eff = a0 * M + a1 * C + K

    for i in range(nt - 1):
        U_pred = U[:, i] + dt * V[:, i] + dt**2 * (0.5 - beta) * A[:, i]
        V_pred = V[:, i] + dt * (1.0 - gamma) * A[:, i]
        P = - (M @ np.ones((n, 1))).flatten() * ag[i + 1]

        u, v = U_pred.copy(), V_pred.copy()
        a = a0 * (u - U[:, i]) - a2 * V[:, i] - a3 * A[:, i]
        ue_iter = ue_prev

        for _ in range(newton_maxit):
            F_h, k_t, ue_new = _bilinear_state(u[0], U[0, i], uy, k0, kp, ue_iter)
            F_iso = F_h + c_iso * v[0]

            R = P - (M @ a + C @ v + K @ u + F_iso * e0)
            Kt = K.copy()
            Kt[0, 0] += k_t
            Keff = M_eff.copy() + (Kt - K)
            Keff[0, 0] += c_iso * a1

            du = np.linalg.solve(Keff, R)
            u += du
            a = a0 * (u - U[:, i]) - a2 * V[:, i] - a3 * A[:, i]
            v = V[:, i] + dt * ((1 - gamma) * A[:, i] + gamma * a)

            if np.linalg.norm(du, ord=np.inf) < newton_tol:
                ue_iter = ue_new
                break
            ue_iter = ue_new

        U[:, i + 1] = u
        V[:, i + 1] = v
        A[:, i + 1] = a

        F_h, _, ue_prev = _bilinear_state(U[0, i + 1], U[0, i], uy, k0, kp, ue_prev)
        F_iso = F_h + c_iso * V[0, i + 1]
        F_iso_hist[i + 1] = F_iso
        F_hyst_hist[i + 1] = F_h
        du0 = U[0, i + 1] - U[0, i]
        E_hyst[i + 1] = E_hyst[i] + 0.5 * (F_hyst_hist[i + 1] + F_hyst_hist[i]) * du0

    return U, V, A, F_iso_hist, F_hyst_hist, E_hyst


# ===============================================================
# === CURVA HISTÉRÉTICA BILINEAL CON MEMORIA (POSTPROCESO) =====
# ===============================================================
def fuerza_bilineal_histeretica(u, k0, kp, Fy):
    """Calcula la curva F–u de un modelo bilineal con memoria."""
    uy = Fy / k0
    F = np.zeros_like(u)
    ue = 0.0
    for i in range(len(u)):
        du = u[i] - (u[i - 1] if i > 0 else 0.0)
        ue = np.clip(ue + du, -uy, uy)
        F[i] = kp * u[i] + (k0 - kp) * ue
    return F


# ====================================================================
# === HISTORIAL DE CORTANTE POR PISO A PARTIR DE ACELERACIONES ======
# ====================================================================
def story_shear_time_history(M, a_abs):
    """Calcula el cortante dinámico por piso V_i(t) = Σ_{j≥i} m_j·a_j(t)."""
    n, nt = a_abs.shape
    m = np.diag(M).reshape(-1, 1)
    F_iner = m * a_abs
    V = np.zeros_like(F_iner)
    V[-1, :] = F_iner[-1, :]
    for i in range(n - 2, -1, -1):
        V[i, :] = V[i + 1, :] + F_iner[i, :]
    Vbase = V[0, :]
    return V, Vbase


# ==============================================================
# === CONVERSIÓN A ACELERACIONES ABSOLUTAS (REL + TERRENO) ====
# ==============================================================
def abs_accel(a_rel, ag_g):
    """Convierte aceleraciones relativas a absolutas (a_abs = a_rel + ag*9.8066500000)."""
    return a_rel + (ag_g * 9.8066500000)[np.newaxis, :]

# ====================================================================
# === HISTORIAL DE CORTANTE POR PISO A PARTIR DE ACELERACIONES =======
# ====================================================================
def story_shear_time_history(M, a_abs):
    """
    Calcula el historial de cortante sísmico por nivel (y la base)
    a partir de las aceleraciones absolutas de cada DOF.
    """
    n, nt = a_abs.shape
    m = np.diag(M).reshape(-1, 1)
    F_iner = m * a_abs
    V = np.zeros_like(F_iner)
    V[-1, :] = F_iner[-1, :]
    for i in range(n - 2, -1, -1):
        V[i, :] = V[i + 1, :] + F_iner[i, :]
    return V, V[0, :]


# ==============================================================
# === CORTANTE MÁXIMO POR NIVEL Y CORTANTE BASAL MÁXIMO =======
# ==============================================================
def peak_shears(V, alturas=None, etiqueta=""):
    """
    Calcula los valores máximos absolutos del cortante por piso
    y del cortante basal a partir del historial V(t).
    """
    Vmax_por_nivel = np.max(np.abs(V), axis=1)
    out = {
        "V_niveles": Vmax_por_nivel,
        "V_base": float(np.max(np.abs(V[0, :]))),
    }
    if alturas is not None:
        out["alturas"] = np.array(alturas, dtype=float)[:len(Vmax_por_nivel)]
    print(f"\n>> Picos de cortante {etiqueta}")
    print(f"   V_base = {out['V_base']:.3f}")
    return out

# =============================================================================
# === MATRIZ DE MASAS CONDENSADA PARA MODELO CON AISLADOR EN LA BASE ==========
# =============================================================================
def calcular_matriz_masas_con_aislador(
    nodes,
    element_node_pairs,
    propiedades,
    peso_especifico=2.4,
    sobrecarga_muerta=0.0,
    b_col_x=0.0
):
    """
    Matriz de masas condensada para modelo con aislador en la base.

    DOF 0   -> base
    DOF 1..n -> masas reales de la superestructura

    Criterio:
    - Columnas: masa repartida 50%-50% entre niveles, incluyendo base y=0.
    - Vigas: peso propio usando longitud libre sin solape con columnas.
    - Carga adicional lineal: usando longitud eje a eje completa.
    """

    import numpy as np
    g = 9.8066500000

    alturas = sorted({round(y, 6) for (x, y, _) in nodes if y > 0})
    n_pisos = len(alturas)

    masas_por_piso = np.zeros(n_pisos + 1, dtype=float)

    node_by_id = {nid: (x, y, nid) for (x, y, nid) in nodes}
    y_to_idx = {y: i for i, y in enumerate(alturas)}

    for n1, n2, tipo in element_node_pairs:
        x1, y1, _ = node_by_id[n1]
        x2, y2, _ = node_by_id[n2]
        A = float(propiedades[tipo]["A"])

        if tipo == "col":
            L = float(np.hypot(x2 - x1, y2 - y1))
            peso_elem = peso_especifico * A * L

            altura_inf = round(min(y1, y2), 6)
            altura_sup = round(max(y1, y2), 6)

            if altura_inf == 0.0:
                masas_por_piso[0] += (peso_elem / 2.0) / g
            elif altura_inf > 0 and altura_inf in y_to_idx:
                masas_por_piso[y_to_idx[altura_inf] + 1] += (peso_elem / 2.0) / g

            if altura_sup > 0 and altura_sup in y_to_idx:
                masas_por_piso[y_to_idx[altura_sup] + 1] += (peso_elem / 2.0) / g

        elif tipo == "viga":
            L = abs(float(x2 - x1))  # longitud eje a eje
            L_real = max(L - 0.5 * b_col_x - 0.5 * b_col_x, 0.0)

            peso_elem = peso_especifico * A * L_real
            altura_viga = round(y1, 6)

            if altura_viga in y_to_idx:
                idx = y_to_idx[altura_viga] + 1
                masas_por_piso[idx] += (peso_elem / g)

                if sobrecarga_muerta > 0.0:
                    peso_sob = sobrecarga_muerta * L
                    masas_por_piso[idx] += (peso_sob / g)

    return np.diag(masas_por_piso)

# =============================================================================
# === CLASE ELEMENTO REDUCIDO LIBRE  (AHORA MISMA LÓGICA QUE FIJO) ============
# =============================================================================
class ElementReducidoLibre:
    """
    MISMA lógica del fijo:
    Elemento viga-columna 2D con 3GDL por nodo [vx, vy, theta]
    (pero con vx tipo diafragma por piso según gdl_map).
    """
    def __init__(self, node_start, node_end, E, I, A, gdl_map):
        import numpy as np
        self.node_start = node_start
        self.node_end   = node_end
        self.E, self.I, self.A = E, I, A

        self.length = np.hypot(node_end[0]-node_start[0], node_end[1]-node_start[1])
        if self.length <= 0:
            raise ValueError("Elemento con longitud nula o negativa.")

        self.k_local  = beam_stiffness_2D(E, I, A, self.length)
        self.T        = transformation_matrix(node_start, node_end)
        self.k_global = self.T.T @ self.k_local @ self.T

        self.dofs = self.assign_dofs(gdl_map)

    def assign_dofs(self, gdl_map):
        """Devuelve [vx_i, vy_i, θ_i, vx_j, vy_j, θ_j] igual que el fijo."""
        i = self.node_start[2]
        j = self.node_end[2]
        dofs_i = [gdl_map.get((i, 'vx')), gdl_map.get((i, 'vy')), gdl_map.get((i, 'theta'))]
        dofs_j = [gdl_map.get((j, 'vx')), gdl_map.get((j, 'vy')), gdl_map.get((j, 'theta'))]
        return dofs_i + dofs_j
    
# =============================================================================
# === FUNCIÓN PARA GENERAR MAPA DE GDL REDUCIDO LIBRE =========================
# =============================================================================
def generar_gdl_map_reducido_libre(nodes):
    """
    Mapa de GDL para modelo aislado:

    - vx: 1 GDL lateral por nivel (diafragma rígido), incluyendo base y=0
    - vy: 1 GDL por nodo solo para y>0
    - theta: 1 GDL por nodo para TODOS los niveles, incluyendo base

    Criterio:
    - En base, vx libre (DOF del sistema aislado)
    - En base, vy fijo
    - En base, theta libre y luego se condensa
    """
    niveles = sorted({round(float(y), 6) for (_, y, _) in nodes})
    if 0.0 in niveles:
        niveles = [0.0] + [y for y in niveles if y != 0.0]
    else:
        niveles = [0.0] + niveles

    # vx compartido por nivel
    vx_por_nivel = {nivel: i for i, nivel in enumerate(niveles)}
    counter = len(niveles)

    gdl_map = {}
    for (x, y, nid) in nodes:
        yk = round(float(y), 6)

        # vx compartido por nivel
        gdl_map[(nid, "vx")] = vx_por_nivel[yk]

        if yk == 0.0:
            # base: vy fijo, theta libre
            gdl_map[(nid, "vy")] = None
            gdl_map[(nid, "theta")] = counter
            counter += 1
        else:
            # niveles superiores: vy y theta libres
            gdl_map[(nid, "vy")] = counter
            counter += 1
            gdl_map[(nid, "theta")] = counter
            counter += 1

    return gdl_map, counter

# =============================================================================
# === ENSAMBLAR MATRIZ GLOBAL (AHORA MISMA LÓGICA QUE FIJO) ===================
# =============================================================================
def assemble_global_stiffness_reducido_libre(elements, total_dofs):
    """
    Ensambla matriz global con 3GDL/nodo (vx, vy, theta), igual que el fijo.
    """
    import numpy as np
    K = np.zeros((int(total_dofs), int(total_dofs)), dtype=float)

    for el in elements:
        ke = el.k_global
        dofs = el.dofs  # [vx_i, vy_i, th_i, vx_j, vy_j, th_j]

        for a in range(6):
            da = dofs[a]
            if da is None:
                continue
            for b in range(6):
                db = dofs[b]
                if db is None:
                    continue
                K[int(da), int(db)] += float(ke[a, b])

    return K

# =============================================================================
# === DETECCIÓN Y EXTRACCIÓN DE REGISTRO SÍSMICO ==============================
# =============================================================================
def detectar_formato_y_extraer(texto):
    """Detecta el formato del registro sísmico (PEER NGA o RENAC/IGP) y extrae datos."""
    lineas = texto.splitlines()

    # --- PEER NGA ---
    for i, linea in enumerate(lineas):
        if "PEER NGA" in linea.upper():
            nombre = lineas[1].split(",")[0].strip()
            unidad = "g" if "UNITS OF G" in lineas[2].upper() else "cm/s²"
            match = re.search(r"NPTS\s*=\s*(\d+),\s*DT\s*=\s*([0-9.Ee+-]+)", lineas[3])
            if match:
                npts = int(match.group(1))
                dt = float(match.group(2))
                datos = extraer_datos(lineas[4:], npts)
                return nombre, unidad, dt, datos

    # --- RENAC / IGP ---
    for i, linea in enumerate(lineas):
        if "RENAC" in linea.upper() or "IGP" in linea.upper():
            nombre, unidad, dt = "", "cm/s²", None
            for l in lineas:
                if "evento:" in l.lower():
                    nombre = l.split(":", 1)[1].strip()
                    break
            for l in lineas:
                if "frecuencia" in l.lower():
                    match = re.search(r"([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", l)
                    if match:
                        frecuencia = float(match.group(1))
                        dt = 1.0 / frecuencia
                    break
            idx_datos = next((j + 1 for j, l in enumerate(lineas) if re.match(r"^_+$", l.strip())), None)
            if idx_datos is None:
                raise ValueError("No se encontró separador de datos en archivo RENAC.")
            datos = extraer_datos(lineas[idx_datos:], None)
            return nombre, unidad, dt, datos

    raise ValueError("Formato de archivo no reconocido.")


# =============================================================================
# === FUNCIÓN: DETECTAR SI HAY QUE CORREGIR O NO EL REGISTRO ==================
# =============================================================================
def detectar_drift(ag, dt, tol_vel=0.02, tol_disp=0.02):
    """
    Detecta drift significativo en un registro sísmico.
    Retorna True si se recomienda corrección.
    """
    from scipy.integrate import cumulative_trapezoid as cumtrapz
    import numpy as np

    t = np.arange(len(ag)) * dt
    vel = cumtrapz(ag, t, initial=0.0)
    desp = cumtrapz(vel, t, initial=0.0)

    vel_ratio  = abs(vel[-1]) / (np.max(np.abs(vel)) + 1e-12)
    disp_ratio = abs(desp[-1]) / (np.max(np.abs(desp)) + 1e-12)

    return (vel_ratio > tol_vel) or (disp_ratio > tol_disp)


# =============================================================================
# === FUNCIÓN: EXTRAER DATOS NUMÉRICOS DE UN REGISTRO SÍSMICO ================
# =============================================================================
def extraer_datos(lineas_datos, npts=None):
    """Extrae los valores numéricos (aceleraciones) desde líneas de texto."""
    datos = []
    for linea in lineas_datos:
        try:
            valores = [float(v) for v in linea.strip().split()]
            datos.extend(valores)
        except ValueError:
            continue
        if npts and len(datos) >= npts:
            break
    return np.array(datos[:npts]) if npts else np.array(datos)


# =============================================================================
# === FUNCIÓN: CORRECCIÓN DE LÍNEA BASE DE UN REGISTRO SÍSMICO ===============
# =============================================================================
def corregir_linea_base(datos, grado=1):
    """Elimina la tendencia (línea base) mediante ajuste polinomial."""
    n = len(datos)
    t = np.arange(n)
    coef = np.polyfit(t, datos, grado)
    polinomio = np.polyval(coef, t)
    return datos - polinomio


# =============================================================================
# === FUNCIÓN: FILTRADO BUTTERWORTH PASA-BANDA ===============================
# =============================================================================
def filtrar_butterworth(datos, dt, f_low=0.075, f_high=25.0, orden=4):
    """Aplica filtro Butterworth pasa-banda a la señal sísmica."""
    fs = 1.0 / dt
    nyq = fs / 2.0
    low, high = f_low / nyq, f_high / nyq
    b, a = signal.butter(orden, [low, high], btype="band")
    return signal.filtfilt(b, a, datos)

# =============================================================================
# === DISEÑO DE AISLADOR LRB AUTOMATICO Y MANUAL ==============================
# === (Keff CONSISTENTE COMO SECANTE REAL DEL BILINEAL) =======================
# =============================================================================
def diseno_aislador_LRB(
    SD1, SDS, T_sin, Mc, nodos_restringidos,
    modo_automatico=True, modo_periodo_objetivo=False, T_objetivo=None,
    *,
    Ku_over_Kd=10.0,   # relación típica Ku/Kd
    LB_factor=0.85     # factor de reducción “LB”
):
    """
    Diseño de aisladores LRB (2D) sin restricciones geométricas impuestas.
    Unidades internas: Tonf, Tonf/m, m, s.
    SD1 y SDS se asumen en 'g'.

    Criterio consistente:
    - La rigidez equivalente usada en el diseño y en el período se toma como
      secante real del bilineal desde el origen hasta D_M.
    - Las propiedades devueltas corresponden a 1 aislador.
    """

    import math
    import numpy as np

    # -------------------- DATOS BASE --------------------
    g = 9.8066500000

    SD1 = float(SD1)
    SDS = float(SDS)

    SM1 = 1.5 * SD1
    T_usar = float(np.atleast_1d(T_sin)[0])

    # ---------------- MASA / PESO ----------------
    if Mc is None or np.ndim(Mc) != 2 or Mc.shape[0] != Mc.shape[1]:
        raise ValueError("Mc debe ser una matriz cuadrada de masas (sistema FIJO).")

    M_super = float(np.sum(np.diag(Mc)))   # [Tonf·s²/m]
    W_total = M_super * g                  # [Tonf]

    n_ais = max(int(len(nodos_restringidos)), 1)
    W_individual = W_total / n_ais

    # ---------------- PROPIEDADES MATERIALES ----------------
    sigma_L = 10.0   # [MPa]
    G = 0.45         # [MPa]
    delta_L = 0.025  # [m]

    sigma_L_tonf = sigma_L * 0.001 * 0.101972 * (1000.0**2)  # [Tonf/m²]
    G_tonf       = G       * 0.001 * 0.101972 * (1000.0**2)  # [Tonf/m²]

    # Tabla empírica B(β)
    betta_vals = np.array([0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50], dtype=float)
    bd_vals    = np.array([0.80, 1.00, 1.20, 1.50, 1.70, 1.90, 2.00], dtype=float)

    def B_of_beta(beta):
        beta = float(max(beta, 0.0))
        if beta <= 0.02:
            return 0.80
        if beta >= 0.50:
            return 2.00
        return float(np.interp(beta, betta_vals, bd_vals))

    def Sa_of_T(T):
        T = float(max(T, 1e-6))
        if SDS <= 0.0:
            return SM1 / T
        Ts = SM1 / SDS
        return SDS if (T <= Ts) else (SM1 / T)

    # ------------------------------------------------------------
    # Flags seguros
    # ------------------------------------------------------------
    T_min_rec = None
    warning_periodo_bajo = False
    mensaje_warning = None

    Q_d_total_LB = None
    K_D_total_LB = None
    k_M = None
    betta_M = None
    D_L = None
    D_B = None
    t_r = None
    D_M = None
    T_final = None
    iteraciones = 0

    # ===================== MODO AUTOMÁTICO =====================
    if modo_automatico and not modo_periodo_objetivo:
        D_L = math.sqrt((0.05 * W_total * 4.0) / (n_ais * math.pi * sigma_L_tonf))
        D_B = 4.0 * D_L
        t_r = D_L

        Sa_seed = Sa_of_T(T_usar)
        D_M = (g * Sa_seed) * (T_usar / (2.0 * math.pi))**2
        D_M = max(D_M, 1e-6)

        tol = 1e-6
        max_iter = 120
        error = 1.0
        iteraciones = 0

        while error > tol and iteraciones < max_iter:
            K_D_total = n_ais * (G_tonf * math.pi * (D_B**2 - D_L**2)) / (4.0 * t_r)
            K_D_total_LB = LB_factor * K_D_total

            Q_d_total = n_ais * (math.pi * D_L**2) * sigma_L_tonf / 4.0
            Q_d_total_LB = LB_factor * Q_d_total

            Kd_total = K_D_total_LB
            Ku_total = max(Ku_over_Kd, 1.01) * Kd_total
            dy_total = Q_d_total_LB / max(Ku_total - Kd_total, 1e-12)

            # ✅ secante real del bilineal en D_M
            if D_M <= dy_total:
                F_M_total = Ku_total * D_M
            else:
                F_M_total = Q_d_total_LB + Kd_total * (D_M - dy_total)

            k_M = F_M_total / max(D_M, 1e-12)

            T_M = 2.0 * math.pi * math.sqrt(W_total / (k_M * g))

            betta_M = (2.0 * Q_d_total_LB * max(D_M - dy_total, 0.0)) / (math.pi * k_M * D_M**2)
            betta_M = max(betta_M, 0.0)

            B_M = B_of_beta(betta_M)

            Sa_Tm = Sa_of_T(T_M)
            D_M_new = (g * Sa_Tm) * (T_M / (2.0 * math.pi))**2 / B_M
            D_M_new = max(D_M_new, 1e-6)

            error = abs(D_M_new - D_M)
            D_M = float(D_M_new)
            iteraciones += 1

        T_final = float(T_M)

    # ================= MODO PERÍODO OBJETIVO ==================
    elif modo_periodo_objetivo and not modo_automatico:
        if T_objetivo is None or float(T_objetivo) <= 0:
            raise ValueError("Debe indicar T_objetivo > 0 cuando modo_periodo_objetivo=True.")

        T_objetivo = float(T_objetivo)

        T_min_rec = 2.0
        warning_periodo_bajo = False
        mensaje_warning = None

        if T_objetivo < 1.5:
            warning_periodo_bajo = True
            mensaje_warning = (
                f"T_objetivo={T_objetivo:.3f}s es MUY bajo para un sistema aislado. "
                "Períodos menores a 1.5 s no representan aislamiento sísmico real, "
                "sino un simple ablandamiento estructural. "
                "Se recomienda usar T_objetivo ≥ 2.0 s."
            )
        elif T_objetivo < 2.0:
            warning_periodo_bajo = True
            mensaje_warning = (
                f"T_objetivo={T_objetivo:.3f}s es bajo para un sistema aislado. "
                "Recomendación técnica: T_objetivo ≥ 2.0 s para lograr "
                "desacople dinámico efectivo."
            )

        D_L = math.sqrt((0.05 * W_total * 4.0) / (n_ais * math.pi * sigma_L_tonf))

        rel_DB_ini = 4.0
        rel_DB_min = 2.5
        margen_tr  = 1.10

        k_M = (4.0 * math.pi**2 * W_total) / (g * T_objetivo**2)

        solucion = False
        for rel_DB in np.linspace(rel_DB_ini, rel_DB_min, num=8):
            D_B = float(rel_DB) * D_L

            Tr_min = (LB_factor * n_ais * (G_tonf * math.pi * (D_B**2 - D_L**2))) / (4.0 * k_M)
            t_r = max(D_L, margen_tr * Tr_min)

            K_D_total = n_ais * (G_tonf * math.pi * (D_B**2 - D_L**2)) / (4.0 * t_r)
            K_D_total_LB = LB_factor * K_D_total

            if k_M > K_D_total_LB:
                solucion = True
                break

        if not solucion:
            D_B_tmp = rel_DB_min * D_L
            K_D_tmp = n_ais * (G_tonf * math.pi * (D_B_tmp**2 - D_L**2)) / (4.0 * D_L)
            K_D_tmp_LB = LB_factor * K_D_tmp
            k_req = K_D_tmp_LB * 1.02
            T_min_posible = 2.0 * math.pi * math.sqrt(W_total / (g * k_req))
            raise ValueError(
                ("Para T_objetivo = {:.3f}s no se logró k_M > K_D_total_LB aun ajustando t_r y D_B/D_L. "
                 "Pruebe con T_objetivo ≥ {:.3f}s, reduzca G o aumente n_ais.")
                .format(T_objetivo, T_min_posible)
            )

        Q_d_total = n_ais * (math.pi * D_L**2) * sigma_L_tonf / 4.0
        Q_d_total_LB = LB_factor * Q_d_total

        Kd_total = K_D_total_LB
        Ku_total = max(Ku_over_Kd, 1.01) * Kd_total
        dy_total = Q_d_total_LB / max(Ku_total - Kd_total, 1e-12)

        # ✅ D_M consistente con secante real del bilineal
        num = Q_d_total_LB - Kd_total * dy_total
        den = max(k_M - Kd_total, 1e-12)
        D_M = num / den
        D_M = max(D_M, dy_total + 1e-9)

        betta_M = (2.0 * Q_d_total_LB * max(D_M - dy_total, 0.0)) / (math.pi * k_M * D_M**2)
        betta_M = max(betta_M, 0.0)

        T_final = float(T_objetivo)
        iteraciones = 0

    else:
        raise ValueError("Activa exactamente uno de los modos: modo_automatico XOR modo_periodo_objetivo.")

    # ---------------- PROPIEDADES POR AISLADOR ----------------
    k_post_1ais = K_D_total_LB / n_ais
    yield_1ais  = Q_d_total_LB / n_ais

    k_inicial_1ais = max(Ku_over_Kd, 1.01) * k_post_1ais
    delta_y = yield_1ais / max((k_inicial_1ais - k_post_1ais), 1e-12)

    D_use = float(max(D_M, 1e-12))
    if D_use <= delta_y:
        F_D = k_inicial_1ais * D_use
    else:
        F_D = yield_1ais + k_post_1ais * (D_use - delta_y)

    keff_1ais = float(F_D / D_use)

    c_1ais = float(betta_M * 2.0 * math.sqrt((keff_1ais * W_individual) / g))
    ratio_postfluencia_1ais = float(k_post_1ais / k_inicial_1ais)

    return {
        "T_M": float(T_final),
        "beta_M": float(betta_M),
        "D_M": float(D_M),
        "k_M": float(k_M),

        "keff_1ais": float(keff_1ais),
        "c_1ais": float(c_1ais),
        "k_inicial_1ais": float(k_inicial_1ais),
        "yield_1ais": float(yield_1ais),
        "k_post_1ais": float(k_post_1ais),
        "ratio_postfluencia_1ais": float(ratio_postfluencia_1ais),

        "D_L": float(D_L),
        "D_B": float(D_B),
        "t_r": float(t_r),

        "delta_L": float(delta_L),
        "delta_y": float(delta_y),

        "iteraciones": int(iteraciones),
        "LB_factor": float(LB_factor),
        "Ku_over_Kd": float(Ku_over_Kd),

        "T_min_rec": None if (T_min_rec is None) else float(T_min_rec),
        "warning_periodo_bajo": bool(warning_periodo_bajo),
        "mensaje_warning": mensaje_warning,
    }

# =============================================================================
# =============== UTILIDADES DE ESTILO + PLOTEO LRB (Tonf / m) ================
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, degrees
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib import rcParams

# ---------- 1) Estilo global: Aquarel Arctic Dark ----------
def set_style_arctic_dark():
    """
    Aplica un tema oscuro tipo 'Aquarel Arctic Dark' a nivel global.
    Llamar una sola vez al inicio del notebook/script.
    """
    plt.style.use('dark_background')
    rcParams.update({
        'figure.facecolor': '#2B3141',
        'axes.facecolor':   '#2B3141',
        'axes.edgecolor':   '#E1E6F0',
        'axes.labelcolor':  '#E8EDF2',
        'xtick.color':      '#DDE3EC',
        'ytick.color':      '#DDE3EC',
        'text.color':       '#E8EDF2',
        'grid.color':       '#5B657A',
        'grid.linestyle':   ':',
        'grid.linewidth':   0.8,
        'axes.linewidth':   1.1,
        'font.family':      'DejaVu Sans',
        'font.size':        10.5,
        'axes.titlesize':   13.5,
        'axes.labelsize':   11.5,
        'legend.facecolor': '#3A4050',
        'legend.edgecolor': '#A7B1C5',
        'legend.fontsize':  10.0,
        'legend.framealpha':0.9,
        'axes.titlepad':    10,
        'lines.linewidth':  2.4
    })

# ---------- Paleta coherente con el tema ----------
COLOR_KE   = '#A8D5FF'  # azul pastel
COLOR_KP   = '#F2A6A0'  # coral pastel
COLOR_KEFF = '#FFE6A3'  # amarillo pastel
COLOR_FILL = '#8FBFEF'  # relleno suave
COLOR_TEXT = '#EDEFF5'
COLOR_BOX  = '#3B4253'
COLOR_EDGE = '#A7B1C5'
GRID_ZERO  = '#7A8498'
HALO = [pe.withStroke(linewidth=3.2, foreground='#2B3141'), pe.Normal()]

# ---------- 2) Modelo bilineal cerrado (rombo LRB) ----------
def rombo_LRB(Ke, Kp, Fy, dy, D2):
    """
    Devuelve los vértices del ciclo bilineal cerrado LRB y magnitudes clave.
    Unidades: Tonf/m, Tonf, m
    """
    if abs(Fy - Ke * dy) > 1e-9:
        dy = Fy / Ke
    F2   = Fy + Kp * (D2 - dy)
    Keff = F2 / D2

    # Cierre simétrico del ciclo
    u3 = (-Fy + Kp*dy - F2 + Ke*D2) / (Ke - Kp)
    f3 = F2 - Ke*(D2 - u3)
    u5 = ( Fy - Kp*dy + F2 - Ke*D2) / (Ke - Kp)
    f5 = -F2 + Ke*(u5 + D2)

    V = np.array([
        [0.0,  0.0],  # 0
        [dy,   Fy ],  # 1  Ke
        [D2,   F2 ],  # 2  Kp
        [u3,   f3 ],  # 3  Ke
        [-D2, -F2],   # 4  Kp
        [u5,   f5 ],  # 5  Ke
        [dy,   Fy ]   # 6  cierre
    ], float)
    return V, F2, Keff, dy

# ---------- 3) Funciones auxiliares para anotaciones ----------
def angle_in_display(ax, x0, y0, x1, y1):
    """Ángulo del segmento en píxeles (versión coherente con unidades en metros)."""
    X0, Y0 = ax.transData.transform((x0, y0))
    X1, Y1 = ax.transData.transform((x1, y1))
    ang = degrees(atan2(Y1 - Y0, X1 - X0))
    return ang - 180 if (ang > 90 or ang < -90) else ang


def put_text_on_segment(ax, P0, P1, text, offset_pts=10, color='white', fontsize=10):
    """
    Coloca `text` paralelo al segmento P0→P1 con un pequeño offset perpendicular.
    Compatible con unidades en metros (sin multiplicar por 1000).
    """
    x0, y0 = P0; x1, y1 = P1
    xm, ym = (x0 + x1)/2, (y0 + y1)/2
    ang = angle_in_display(ax, x0, y0, x1, y1)

    (X0, Y0) = ax.transData.transform((x0, y0))
    (X1, Y1) = ax.transData.transform((x1, y1))
    dxs, dys = (X1 - X0), (Y1 - Y0)
    L = max(1e-12, np.hypot(dxs, dys))
    nx, ny = -dys/L, dxs/L
    dpx, dpy = nx * offset_pts, ny * offset_pts

    ax.annotate(text, xy=(xm, ym), xytext=(dpx, dpy),
                textcoords='offset points', ha='center', va='center',
                fontsize=fontsize, color=color,
                rotation=ang, rotation_mode='anchor',
                path_effects=HALO, clip_on=True)


def box_vertex_outside(ax, V, idx, text, fs=9, base_px=42, step_px=4):
    """
    Caja 'por fuera' del rombo, alineada con la bisectriz exterior en pantalla.
    Adaptada para coordenadas en metros (sin multiplicar por 1000).
    """
    n = len(V) - 1
    i = idx % n
    Pi, Pp, Pn = V[i], V[(i-1) % n], V[(i+1) % n]
    Xi, Yi = ax.transData.transform((Pi[0], Pi[1]))
    Xp, Yp = ax.transData.transform((Pp[0], Pp[1]))
    Xn, Yn = ax.transData.transform((Pn[0], Pn[1]))

    def unit(v):
        L = np.hypot(v[0], v[1])
        return v / (L if L > 1e-12 else 1)

    u1, u2 = unit([Xi - Xp, Yi - Yp]), unit([Xn - Xi, Yn - Yi])

    Cx = np.mean(V[:-1, 0])
    Cy = np.mean(V[:-1, 1])
    Cxs, Cys = ax.transData.transform((Cx, Cy))
    vc = np.array([Cxs - Xi, Cys - Yi])

    def outward_normal(u):
        nl, nr = np.array([-u[1], u[0]]), np.array([u[1], -u[0]])
        return nl if np.dot(nl, vc) < np.dot(nr, vc) else nr

    n1, n2 = outward_normal(u1), outward_normal(u2)
    bis = unit(n1 / np.hypot(*n1) + n2 / np.hypot(*n2))

    poly = Path(np.column_stack((V[:-1, 0], V[:-1, 1])), closed=True)
    k = base_px
    for _ in range(80):
        Xb, Yb = Xi + bis[0] * k, Yi + bis[1] * k
        xb, yb = ax.transData.inverted().transform((Xb, Yb))
        if not poly.contains_point((xb, yb)):
            break
        k += step_px

    ax.annotate(text, xy=(Pi[0], Pi[1]), xytext=(xb, yb),
                textcoords='data', ha='center', va='center', fontsize=fs,
                bbox=dict(boxstyle="round,pad=0.25", fc=COLOR_BOX, ec=COLOR_EDGE, alpha=0.9),
                arrowprops=dict(arrowstyle='->', lw=0.8, color='#C3C9D8'),
                color=COLOR_TEXT, clip_on=True)

# ---------- 4) Función principal de ploteo ----------
def plot_ciclo_histeretico_lrb(
    Ke, Kp, Fy, dy, D2, Keff_ref=None,
    titulo=None,
    xlabel=None,
    ylabel=None,
    *, savepath=None, show=False
):
    """
    Dibuja ciclo bilineal LRB (Tonf/m, Tonf, m).
    NO depende de tr() ni de T. Los textos se pasan desde app.py.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    V, F2_calc, Keff_calc, dy = rombo_LRB(Ke, Kp, Fy, dy, D2)
    Keff_usar = Keff_ref if Keff_ref is not None else Keff_calc

    # ✅ fuerza secante consistente con Keff_usar
    F2_sec = Keff_usar * D2

    fig, ax = plt.subplots(figsize=(9, 7.75))
    ax.fill(V[:, 0], V[:, 1], color=COLOR_FILL, alpha=0.10)

    tramos = [
        (V[0], V[1], COLOR_KE),
        (V[1], V[2], COLOR_KP),
        (V[2], V[3], COLOR_KE),
        (V[3], V[4], COLOR_KP),
        (V[4], V[5], COLOR_KE),
        (V[5], V[6], COLOR_KP),
    ]
    for (P0, P1, col) in tramos:
        ax.plot([P0[0], P1[0]], [P0[1], P1[1]], color=col, lw=2.4)

    ax.plot([0, D2], [0, F2_sec], "--", lw=2.0, color=COLOR_KEFF)

    fig.canvas.draw()

    put_text_on_segment(ax, V[0], V[1], "Ke", color=COLOR_KE)
    put_text_on_segment(ax, V[1], V[2], "Kp", color=COLOR_KP)
    put_text_on_segment(ax, V[2], V[3], "Ke", color=COLOR_KE)
    put_text_on_segment(ax, V[3], V[4], "Kp", color=COLOR_KP)
    put_text_on_segment(ax, V[4], V[5], "Ke", color=COLOR_KE)
    put_text_on_segment(ax, V[5], V[6], "Kp", color=COLOR_KP)
    put_text_on_segment(ax, (0, 0), (D2, F2_sec), "Keff", color=COLOR_KEFF)

    xmax_abs = np.max(np.abs(V[:, 0]))
    ymax_abs = np.max(np.abs(V[:, 1]))
    ax.set_xlim(-xmax_abs * 1.35, xmax_abs * 1.35)
    ax.set_ylim(-ymax_abs * 1.30, ymax_abs * 1.30)

    box_vertex_outside(ax, V, 1, f"Δ₁ = {dy:.4f} m\nF₁ = {Fy:.3f} Tonf")
    box_vertex_outside(ax, V, 2, f"Δ₂ = {D2:.4f} m\nF₂ = {F2_sec:.3f} Tonf")
    box_vertex_outside(ax, V, 4, f"−Δ₂ = {D2:.4f} m\n−F₂ = {F2_sec:.3f} Tonf")

    handles = [
        Line2D([0], [0], color=COLOR_KE,   lw=2.4, label=f"Ke = {Ke:.3f} Tonf/m"),
        Line2D([0], [0], color=COLOR_KP,   lw=2.4, label=f"Kp = {Kp:.3f} Tonf/m"),
        Line2D([0], [0], color=COLOR_KEFF, lw=2.0, linestyle="--",
               label=f"Keff = {Keff_usar:.3f} Tonf/m"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9)
    for t in leg.get_texts():
        t.set_color(COLOR_TEXT)

    ax.axhline(0, color=GRID_ZERO, lw=1.1)
    ax.axvline(0, color=GRID_ZERO, lw=1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Textos (ya traducidos desde app.py)
    if titulo is not None:
        ax.set_title(titulo, fontweight="bold", color="#F4F6FA")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.grid(True, alpha=0.45)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300)
    if show:
        plt.show()

    return fig, ax

def plot_lrb_from_resultados(
    resultados_ais, *,
    titulo=None, xlabel=None, ylabel=None,
    savepath=None, show=False
):
    """
    Wrapper: toma resultados_ais (Tonf/m, Tonf, m) y grafica el ciclo.
    dy = delta_L (correcto). c_1ais es amortiguamiento viscoso, NO dy.
    """
    import numpy as np

    Ke   = float(resultados_ais["k_inicial_1ais"])
    Kp   = float(resultados_ais["k_post_1ais"])
    Fy   = float(resultados_ais["yield_1ais"])
    dy   = float(resultados_ais.get("delta_y", resultados_ais["delta_L"]))
    D2   = float(resultados_ais["D_M"])
    Keff = float(resultados_ais.get("keff_1ais", np.nan))

    Keff_ref = None if (not np.isfinite(Keff)) else Keff

    return plot_ciclo_histeretico_lrb(
        Ke, Kp, Fy, dy, D2,
        Keff_ref=Keff_ref,
        titulo=titulo,
        xlabel=xlabel,
        ylabel=ylabel,
        savepath=savepath,
        show=show
    )

# =============================================================================
# === HELPERS BLOQUE 1: PARÁMETROS GENERALES DEL MODELO =======================
# =============================================================================

def seccion_rectangular_cm_to_SI(b_cm: float, h_cm: float) -> tuple[float, float]:
    """
    Sección rectangular b×h en cm -> (A [m²], I [m⁴]) con I = b*h^3/12.
    """
    b_m = float(b_cm) / 100.0
    h_m = float(h_cm) / 100.0
    A = b_m * h_m
    I = b_m * (h_m ** 3) / 12.0
    return float(A), float(I)


def seccion_AI_cm_to_SI(A_cm2: float, I_cm4: float) -> tuple[float, float]:
    """
    (A [cm²], I [cm⁴]) -> (A [m²], I [m⁴]).
    """
    A = float(A_cm2) / 1e4
    I = float(I_cm4) / 1e8
    return float(A), float(I)


def build_param_estruct(
    n_pisos: int,
    n_vanos: int,
    l_vano: float,
    h_piso_1: float,
    h_piso_restantes: float,
    E: float,
    A_col: float,
    I_col: float,
    A_viga: float,
    I_viga: float,
    peso_especifico: float,
    sobrecarga_muerta: float,
    amortiguamiento_ratio: float,
    modo_avanzado: bool,
    b_col_cm=None, h_col_cm=None, b_viga_cm=None, h_viga_cm=None,
    A_col_cm2=None, I_col_cm4=None, A_viga_cm2=None, I_viga_cm4=None
) -> dict:
    """
    Arma el dict param_estruct de forma consistente.
    amortiguamiento_ratio: ζ en fracción (ej: 0.05)
    """
    return {
        "n_pisos": int(n_pisos),
        "n_vanos": int(n_vanos),
        "l_vano": float(l_vano),
        "h_piso_1": float(h_piso_1),
        "h_piso_restantes": float(h_piso_restantes),

        "E": float(E),
        "I_col": float(I_col),
        "A_col": float(A_col),
        "I_viga": float(I_viga),
        "A_viga": float(A_viga),

        "peso_especifico": float(peso_especifico),
        "sobrecarga_muerta": float(sobrecarga_muerta),
        "amortiguamiento": float(amortiguamiento_ratio),

        "modo_avanzado": bool(modo_avanzado),
        "b_col_cm": b_col_cm,
        "h_col_cm": h_col_cm,
        "b_viga_cm": b_viga_cm,
        "h_viga_cm": h_viga_cm,
        "A_col_cm2": A_col_cm2 if modo_avanzado else None,
        "I_col_cm4": I_col_cm4 if modo_avanzado else None,
        "A_viga_cm2": A_viga_cm2 if modo_avanzado else None,
        "I_viga_cm4": I_viga_cm4 if modo_avanzado else None,
    }

# =============================================================================
# === BLOQUE 2 (MOTOR): MODELO + MASAS + TRANSFORMACIÓN + CONDENSACIÓN =========
# =============================================================================
import numpy as np
from numpy.linalg import inv

def b2_get_params_from_param_estruct(param_estruct: dict) -> dict:
    """
    Replica tu _get_params() pero sin Streamlit.
    """
    params = param_estruct or {}
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
        "modo_avanzado": bool(params.get("modo_avanzado", False)),
        "b_col_cm": params.get("b_col_cm", None),
    }

def b2_params_key(p: dict) -> tuple:
    """
    Replica tu _params_key().
    """
    return (
        p["n_pisos"],
        p["n_vanos"],
        p["l_vano"],
        p["h_piso_1"],
        p["h_piso_restantes"],
        p["E"],
        p["I_col"],
        p["A_col"],
        p["I_viga"],
        p["A_viga"],
        p["peso_especifico"],
        p["sobrecarga_muerta"],
        p["amortiguamiento"],
        p.get("modo_avanzado", False),
        p.get("b_col_cm", None),
    )

def b2_generar_modelo(
    p: dict,
    Element,
    generar_gdl_map,
    assemble_global_stiffness,
    calcular_matriz_masas_por_piso=None,
    warn_callback=None,
):
    """
    Ejecuta TU misma secuencia:
    1) nodes
    2) gdl_map
    3) propiedades
    4) conectividades
    5) elements + K_global
    6) M_cond (si función existe)
    7) T_trans
    8) K_cond + k_modelo/k_aprox/ratio
    Devuelve dict con todo lo que guardabas en session_state.
    """

    def _warn(msg: str):
        if callable(warn_callback):
            warn_callback(msg)

    n_pisos, n_vanos = p["n_pisos"], p["n_vanos"]
    l_vano = p["l_vano"]
    h_piso_1, h_piso_restantes = p["h_piso_1"], p["h_piso_restantes"]
    E, I_col, A_col = p["E"], p["I_col"], p["A_col"]
    I_viga, A_viga = p["I_viga"], p["A_viga"]
    peso_especifico = p["peso_especifico"]
    sobrecarga_muerta = p["sobrecarga_muerta"]
    modo_avanzado = p.get("modo_avanzado", False)
    b_col_cm = p.get("b_col_cm", None)

    if (not modo_avanzado) and (b_col_cm is not None):
        b_col_x = float(b_col_cm) / 100.0
    else:
        b_col_x = 0.0

    # 1) Nodos
    nodes = []
    y_actual = 0.0
    for i in range(n_pisos + 1):
        base_id = i * (n_vanos + 1)
        for j in range(n_vanos + 1):
            nodes.append((j * l_vano, y_actual, base_id + j))
        y_actual += h_piso_1 if i == 0 else h_piso_restantes

    # 2) GDL map
    gdl_map = generar_gdl_map(nodes)

    # 3) Propiedades
    propiedades = {
        "col": {"E": E, "I": I_col, "A": A_col},
        "viga": {"E": E, "I": I_viga, "A": A_viga},
    }

    # 4) Conectividades
    element_node_pairs = []
    for i in range(n_pisos):
        row_i = i * (n_vanos + 1)
        row_j = (i + 1) * (n_vanos + 1)
        for j in range(n_vanos + 1):
            element_node_pairs.append((row_i + j, row_j + j, "col"))

    for i in range(1, n_pisos + 1):
        row_i = i * (n_vanos + 1)
        for j in range(n_vanos):
            element_node_pairs.append((row_i + j, row_i + j + 1, "viga"))

    # 5) Ensamble global
    elements = []
    for n1, n2, tipo in element_node_pairs:
        node_i = nodes[n1]
        node_j = nodes[n2]
        prop = propiedades[tipo]
        elements.append(Element(node_i, node_j, prop["E"], prop["I"], prop["A"], gdl_map))

    total_dofs = max(dof for el in elements for dof in el.dofs if dof is not None) + 1
    K_global = assemble_global_stiffness(elements, total_dofs)

    # 6) Masa condensada por piso
    M_cond = None
    if callable(calcular_matriz_masas_por_piso):
        M_cond = calcular_matriz_masas_por_piso(
            nodes,
            element_node_pairs,
            propiedades,
            peso_especifico=peso_especifico,
            sobrecarga_muerta=sobrecarga_muerta,
            b_col_x=b_col_x,
        )

    # 7) Matriz de transformación
    T_trans = None
    if M_cond is not None:
        nodes_arr = np.array(nodes, dtype=float)
        alturas = np.unique(nodes_arr[:, 1])
        pisos_y_emp = [round(float(y), 6) for y in alturas if float(y) > 0.0]
        altura_a_col = {y: i for i, y in enumerate(pisos_y_emp)}

        T_trans = np.zeros((K_global.shape[0], len(pisos_y_emp)))
        for (x, y, nid) in nodes:
            if y > 0:
                dof_vx = gdl_map.get((nid, "vx"))
                if dof_vx is not None:
                    T_trans[int(dof_vx), altura_a_col[round(float(y), 6)]] = 1.0

    # 8) Diafragma rígido + condensación
    K_cond = None
    k_modelo = None
    k_aprox = None
    ratio_k = None

    nodes_arr = np.array(nodes, dtype=float)
    alturas = np.unique(nodes_arr[:, 1])
    pisos_y = [round(float(y), 6) for y in alturas if float(y) > 0.0]
    n_pisos_cond = len(pisos_y)

    if n_pisos_cond == 0:
        raise ValueError("No hay pisos (y>0).")

    master_node_por_piso = {}
    for y in pisos_y:
        nodos_en_y = [(float(x), int(nid)) for (x, yy, nid) in nodes if round(float(yy), 6) == y]
        nodos_en_y.sort(key=lambda t: t[0])
        master_node_por_piso[y] = nodos_en_y[0][1]

    dofUx_piso = []
    for y in pisos_y:
        nid_m = master_node_por_piso[y]
        dof_m = gdl_map.get((nid_m, "vx"))
        if dof_m is None:
            raise ValueError(f"Master sin vx en y={y}. Revisa gdl_map.")
        dofUx_piso.append(int(dof_m))

    vx_all = [int(dof) for (nid, tipo), dof in gdl_map.items() if tipo == "vx" and dof is not None]
    vy_all = [int(dof) for (nid, tipo), dof in gdl_map.items() if tipo == "vy" and dof is not None]
    th_all = [int(dof) for (nid, tipo), dof in gdl_map.items() if tipo == "theta" and dof is not None]

    set_masters = set(dofUx_piso)
    vx_slaves = sorted([d for d in vx_all if d not in set_masters])
    gdl_ss = sorted(vx_slaves + vy_all + th_all)

    n_full = K_global.shape[0]
    n_p = n_pisos_cond
    n_s = len(gdl_ss)

    s_col = {dof: j for j, dof in enumerate(gdl_ss)}

    piso_index = {y: i for i, y in enumerate(pisos_y)}
    dof_to_piso = {}
    for (x, yy, nid) in nodes:
        yk = round(float(yy), 6)
        if yk in piso_index:
            dof_vx = gdl_map.get((nid, "vx"))
            if dof_vx is not None:
                dof_to_piso[int(dof_vx)] = piso_index[yk]

    R = np.zeros((n_full, n_p + n_s), dtype=float)

    for dof_vx in vx_all:
        piso_idx = dof_to_piso.get(int(dof_vx))
        if piso_idx is not None:
            R[int(dof_vx), piso_idx] = 1.0

    for dof in gdl_ss:
        R[int(dof), n_p + s_col[int(dof)]] = 1.0

    Kd = R.T @ K_global @ R
    Kd = 0.5 * (Kd + Kd.T)

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
            _warn("Kss singular; se usó pseudo-inversa (pinv).")
        K_cond = Kpp - Kps @ Kss_inv @ Ksp

    K_cond = 0.5 * (K_cond + K_cond.T)
    K_cond = np.array(K_cond, dtype=float, copy=True)

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

    model_summary = {
        "n_nodes": len(nodes),
        "n_elems": len(elements),
        "n_dofs": int(total_dofs),
    }

    return {
        "nodes": nodes,
        "element_node_pairs": element_node_pairs,
        "propiedades": propiedades,
        "gdl_map": gdl_map,
        "elements": elements,
        "K_global": K_global,
        "M_cond": M_cond,
        "T_trans": T_trans,
        "K_cond": K_cond,
        "k_modelo": k_modelo,
        "k_aprox": k_aprox,
        "ratio_k": ratio_k,
        "model_summary": model_summary,
    }

# =============================================================================
# FUNCIONES - BLOQUE 3: NEC-24 + Registro + Espectro del registro + Escalamiento
# (SIN STREAMLIT UI)
# =============================================================================
import os
import numpy as np
import chardet
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy import signal

# =========================
# TABLAS NEC-24 (constantes)
# =========================
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

# g estándar (consistente en todo el proyecto)
G_STD = 9.8066500000

def nec24_espectro(z: float, zona: str, suelo: str, R: float, r: float = 1.0,
                   T_final: float = 5.0, delta_t: float = 0.01):
    """
    Devuelve:
      T, Sa_elast[g], Sa_inelas[g], SDS[g], SD1[g*s], Fa, Fd, Fs
    """
    zona = str(zona).upper().strip()
    suelo = str(suelo).upper().strip()
    if zona not in ZONAS_DICT:
        raise ValueError(f"Zona inválida: {zona}")
    if suelo not in TABLA_FA:
        raise ValueError(f"Suelo inválido: {suelo}")

    idx_zona = ZONAS_DICT[zona]
    Fa = float(TABLA_FA[suelo][idx_zona])
    Fd = float(TABLA_FD[suelo][idx_zona])
    Fs = float(TABLA_FS[suelo][idx_zona])

    T0 = 0.1 * Fs * Fd / Fa
    Tc = 0.45 * Fs * Fd / Fa
    TL = 2.4 * Fd

    T = np.linspace(0.0, float(T_final), int(float(T_final) / float(delta_t)) + 1)
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

    SDS = 2.4 * z * Fa

    # SD1 = Sa(T=1s)
    T_SD1 = 1.0
    if T_SD1 < T0:
        SD1 = z * Fa * (1 + 1.4 * (T_SD1 / T0))
    elif T0 <= T_SD1 < Tc:
        SD1 = 2.4 * z * Fa
    elif Tc <= T_SD1 < TL:
        SD1 = 2.4 * z * Fa * (Tc / T_SD1) ** r
    else:
        SD1 = 2.4 * z * Fa * (Tc / TL) ** r * (TL / T_SD1) ** 2

    return T, Sa_elast, Sa_inelas, SDS, SD1, Fa, Fd, Fs

def detectar_fuente(texto: str) -> str:
    t = (texto or "").lower()
    if ("pacific earthquake engineering research" in t) or ("peer strong motion" in t) or ("ngawest" in t) or (".at2" in t):
        return "PEER NGA"
    if ("red nacional de acelerógrafos" in t) or ("renac" in t) or ("igepn" in t):
        return "RENAC (IG-EPN)"
    if ("instituto geofísico" in t) and ("pucp" not in t):
        return "IGP"
    return "Desconocido"


def leer_archivo_bytes_a_texto(file_bytes: bytes) -> str:
    enc = chardet.detect(file_bytes).get("encoding") or "utf-8"
    return file_bytes.decode(enc, errors="ignore")


def cargar_ejemplo_desde_carpeta(filename: str = "EJEMPLO.txt", base_dir: str | None = None):
    """
    Devuelve: nombre, unidad, dt, ag_mps2, fuente, texto_crudo
    Requiere que exista detectar_formato_y_extraer(texto) en tu proyecto (NO se define aquí).
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {filename} en: {base_dir}")

    with open(path, "rb") as f:
        raw = f.read()

    texto = leer_archivo_bytes_a_texto(raw)
    fuente = detectar_fuente(texto)

    # detecta formato (tu función existente)
    nombre, unidad, dt, ag = detectar_formato_y_extraer(texto)  # noqa: F821
    ag = np.asarray(ag, dtype=float).ravel()
    dt = float(dt)

    if unidad == "cm/s²":
        ag_mps2 = ag / 100.0
    elif unidad == "g":
        ag_mps2 = ag * G_STD
    elif unidad == "m/s²":
        ag_mps2 = ag
    else:
        raise ValueError(f"Unidad no reconocida: {unidad}")

    return str(nombre), str(unidad), dt, np.asarray(ag_mps2, float), str(fuente), texto

def procesar_registro(ag_mps2: np.ndarray, dt: float, aplicar_proc: bool):
    """
    Devuelve dict:
      t, ag_orig, vel_orig, disp_orig,
      ag_proc, vel_proc, disp_proc,
      ag_base (proc si aplica, sino original)
    """
    ag_orig = np.asarray(ag_mps2, dtype=float).ravel()
    dt = float(dt)

    t = np.linspace(0.0, dt * (len(ag_orig) - 1), len(ag_orig))

    # ORIGINAL
    vel_orig = cumtrapz(ag_orig, t, initial=0.0)
    disp_orig = cumtrapz(vel_orig, t, initial=0.0)

    ag_proc = None
    vel_proc = None
    disp_proc = None

    if aplicar_proc:
        # 1) baseline correction
        ag_bc = signal.detrend(ag_orig, type="linear")

        # 2) bandpass filter
        fs = 1.0 / dt
        fnyquist = 0.5 * fs

        low = 0.10 / fnyquist
        high = min(25.0 / fnyquist, 0.999)

        if low <= 0 or low >= high:
            ag_filt = ag_bc.copy()
        else:
            b_bandpass, a_bandpass = signal.butter(4, [low, high], btype="bandpass")
            ag_filt = signal.lfilter(b_bandpass, a_bandpass, ag_bc)

        # 3) velocity
        vel_raw = cumtrapz(ag_filt, t, initial=0.0)

        # 4) velocity baseline correction
        coef_v = np.polyfit(t, vel_raw, 1)
        vel_proc = vel_raw - np.polyval(coef_v, t)

        # 5) displacement
        disp_raw = cumtrapz(vel_proc, t, initial=0.0)

        # 6) displacement correction
        coef_u = np.polyfit(t, disp_raw, 2)
        disp_proc = disp_raw - np.polyval(coef_u, t)

        ag_proc = ag_filt

    ag_base = ag_proc if (aplicar_proc and ag_proc is not None) else ag_orig

    return {
        "t": t,
        "ag_orig": ag_orig,
        "vel_orig": vel_orig,
        "disp_orig": disp_orig,
        "ag_proc": ag_proc,
        "vel_proc": vel_proc,
        "disp_proc": disp_proc,
        "ag_base": ag_base,
    }
    
def response_spectrum_newmark(ag_mps2: np.ndarray, dt: float, T: np.ndarray, xi: float = 0.05):
    """
    Espectro (PSA) en g. (tu mismo Newmark)
    """
    ag = np.asarray(ag_mps2, dtype=float).ravel()
    T = np.asarray(T, dtype=float).ravel()
    T = np.clip(T, 1e-4, None)

    beta = 1/4
    gamma = 1/2
    m = 1.0
    n = len(ag)

    Sa_g = np.zeros_like(T, dtype=float)

    for i, Ti in enumerate(T):
        w = 2*np.pi / Ti
        k = m * w**2
        c = 2 * xi * m * w

        u = 0.0
        ud = 0.0
        udd = 0.0

        a0 = 1.0/(beta*dt*dt)
        a1 = gamma/(beta*dt)
        a2 = 1.0/(beta*dt)
        a3 = 1.0/(2*beta) - 1.0
        a4 = gamma/beta - 1.0
        a5 = dt*(gamma/(2*beta) - 1.0)

        k_eff = k + a0*m + a1*c
        u_max = 0.0

        for j in range(n):
            p = -m * ag[j]
            p_eff = (
                p
                + m*(a0*u + a2*ud + a3*udd)
                + c*(a1*u + a4*ud + a5*udd)
            )

            u_new = p_eff / k_eff
            udd_new = a0*(u_new - u) - a2*ud - a3*udd
            ud_new  = ud + dt*((1-gamma)*udd + gamma*udd_new)

            u, ud, udd = u_new, ud_new, udd_new
            if abs(u) > u_max:
                u_max = abs(u)

        Sa_g[i] = (w**2 * u_max) / G_STD

    return Sa_g


def lsq_scale_factor(Sa_reg: np.ndarray, Sa_target: np.ndarray):
    """
    Factor de escala por mínimos cuadrados en espacio logarítmico.
    Más robusto que el LSQ lineal simple y evita sobreescalado excesivo.
    """
    Sa_reg = np.asarray(Sa_reg, dtype=float).ravel()
    Sa_target = np.asarray(Sa_target, dtype=float).ravel()

    eps = 1e-12
    ok = (
        np.isfinite(Sa_reg) & np.isfinite(Sa_target) &
        (Sa_reg > eps) & (Sa_target > eps)
    )

    if np.count_nonzero(ok) < 3:
        return 1.0

    log_ratio = np.log(Sa_target[ok]) - np.log(Sa_reg[ok])
    SF = float(np.exp(np.mean(log_ratio)))

    if (not np.isfinite(SF)) or (SF <= 0):
        return 1.0

    return SF

def make_T_rs_piecewise(Tmin: float = 0.05, Tmax: float = 5.0):
    Tmax = float(max(Tmin + 1e-6, Tmax))
    T0 = np.linspace(Tmin, min(0.35, Tmax), 420)
    T1 = np.linspace(min(0.35, Tmax), min(0.50, Tmax), 180)
    T2 = np.linspace(min(0.50, Tmax), min(2.00, Tmax), 200)
    T3 = np.linspace(min(2.00, Tmax), Tmax, 140)
    return np.unique(np.concatenate([T0, T1, T2, T3]))

def decimate_adaptive(ag: np.ndarray, dt: float, dt_target: float):
    dt = float(dt)
    dt_target = float(dt_target)
    if dt <= dt_target:
        dec = int(max(1, round(dt_target / dt)))
    else:
        dec = 1
    return ag[::dec], dt * dec, dec


def compute_Sa_piecewise(ag_base: np.ndarray, dt: float, T_rs: np.ndarray, xi: float):
    """
    MISMA IDEA que tu versión: calcula Sa por tramos con dt_target distinto.
    (sin cache aquí; el cache lo haces en app.py si quieres)
    """
    T_rs = np.asarray(T_rs, dtype=float).ravel()
    Sa = np.zeros_like(T_rs, dtype=float)

    m0 = (T_rs <= 0.25)
    m1 = (T_rs > 0.25) & (T_rs <= 0.50)
    m2 = (T_rs > 0.50) & (T_rs <= 2.00)
    m3 = (T_rs > 2.00)

    if np.any(m0):
        dt_t0 = max(float(dt), 0.005)
        ag0, dt0, _ = decimate_adaptive(ag_base, dt, dt_target=dt_t0)
        Sa[m0] = response_spectrum_newmark(ag0, float(dt0), T_rs[m0], xi=float(xi))

    if np.any(m1):
        dt_t1 = max(float(dt), 0.010)
        ag1, dt1, _ = decimate_adaptive(ag_base, dt, dt_target=dt_t1)
        Sa[m1] = response_spectrum_newmark(ag1, float(dt1), T_rs[m1], xi=float(xi))

    if np.any(m2):
        dt_t2 = max(float(dt), 0.020)
        ag2, dt2, _ = decimate_adaptive(ag_base, dt, dt_target=dt_t2)
        Sa[m2] = response_spectrum_newmark(ag2, float(dt2), T_rs[m2], xi=float(xi))

    if np.any(m3):
        dt_t3 = max(float(dt), 0.035)
        ag3, dt3, _ = decimate_adaptive(ag_base, dt, dt_target=dt_t3)
        Sa[m3] = response_spectrum_newmark(ag3, float(dt3), T_rs[m3], xi=float(xi))

    return Sa

# =============================================================================
# === BLOQUE 4 HELPERS: cache modal key + checks bilinear beta ================
# =============================================================================
import numpy as np
import math

def _km_key(K, M) -> tuple:
    K = np.asarray(K); M = np.asarray(M)
    return (K.shape, M.shape, float(np.sum(K)), float(np.sum(M)))

def _beta_from_bilinear_cycle(Fy: float, dy: float, D: float, Keff: float) -> float:
    """
    β_eq ≈ Ed / (4π Es)
    Ed ≈ 4 Fy (D - dy)
    Es = 0.5 Keff D^2
    => β ≈ Ed / (2π Keff D^2)
    """
    D = float(D)
    Keff = float(Keff)
    if (D <= 0) or (Keff <= 0):
        return float("nan")
    dy_use = max(0.0, min(float(dy), D))
    Ed = 4.0 * float(Fy) * max(0.0, (D - dy_use))
    beta = Ed / (2.0 * math.pi * Keff * (D**2))
    return max(0.0, beta)

def _compute_checks(resultados_ais: dict) -> dict:
    Ke   = float(resultados_ais["k_inicial_1ais"])
    Kp   = float(resultados_ais["k_post_1ais"])
    Fy   = float(resultados_ais["yield_1ais"])
    dy   = float(resultados_ais.get("delta_y", np.nan))     # ✅ dy REAL (yield)
    Dm   = float(resultados_ais["D_M"])
    beta = float(resultados_ais.get("beta_M", np.nan))
    Keff = float(resultados_ais.get("keff_1ais", np.nan))

    ok_dy = np.isfinite(dy) and (dy > 0) and (Dm > dy)
    ok_k  = (Ke > Kp > 0)
    ok_b  = np.isfinite(beta) and (0.02 <= beta <= 0.50)

    beta_cycle = _beta_from_bilinear_cycle(
        Fy=Fy,
        dy=dy if np.isfinite(dy) else 0.0,
        D=Dm,
        Keff=Keff
    )
    ok_match = np.isfinite(beta) and np.isfinite(beta_cycle) and (abs(beta - beta_cycle) <= 0.10)

    return {
        "ok_dy": ok_dy,
        "ok_k": ok_k,
        "ok_b": ok_b,
        "ok_match": ok_match,
        "beta": beta,
        "beta_cycle": beta_cycle,
        "dy": dy,
    }

# =============================================================================
# ========= BLOQUE 5 HELPERS: MODAL + ESQUEMAS OPTIMIZADO PARA HASTA 30 PISOS ==
# =============================================================================
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.linalg import eigh

# -----------------------------------------------------------------
# ✅ tr() compatible (NO rompe si no existe T en session_state)
# -----------------------------------------------------------------
def tr(key: str) -> str:
    T_local = st.session_state.get("T", None)
    lang = st.session_state.get("lang", "en")
    if isinstance(T_local, dict):
        return T_local.get(lang, T_local.get("en", {})).get(key, key)
    return key

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
# Utilidades (MISMAS, sin cambiar lógica / nombres)
# -----------------------------------------------------------------
def modal_props(K, M):

    K = np.asarray(K, float)
    M = np.asarray(M, float)

    # Problema generalizado
    w2, V = eigh(K, M)

    idx = np.argsort(w2)
    w2 = w2[idx]
    V  = V[:, idx]

    # Normalización modal respecto a M
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
    ✅ Textos internos bilingües usando tr().
    ✅ Soporta niveles con o sin un punto extra en la base.
    """
    Vn = np.asarray(Vn, dtype=float)
    T  = np.asarray(T, dtype=float).ravel()
    niveles = np.asarray(niveles, dtype=float).ravel()

    # Normalizar por modo
    den = np.max(np.abs(Vn), axis=0)
    den = np.where(den == 0, 1.0, den)
    Vplot = Vn / den

    n_gdl = int(Vplot.shape[0])
    n_modos = int(Vplot.shape[1])

    if n_modos <= 0:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.text(
            0.5, 0.5, tr("b5_no_modes"),
            ha="center", va="center",
            color=COLOR_TEXT, fontsize=12, path_effects=halo2
        )
        ax.axis("off")
        return fig

    ncols = int(max(1, min(int(ncols), n_modos)))
    nrows = int(np.ceil(n_modos / ncols))

    fig_w = 1.75 * ncols
    fig_h = 3.9 * nrows

    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=True)
    fig.patch.set_facecolor(BG)

    axs_list = np.array(axs).ravel().tolist() if isinstance(axs, np.ndarray) else [axs]
    for ax in axs_list:
        ax.set_facecolor(BG)

    fs_title = 7.5
    mode_lbl = tr("b5_mode_lbl")

    for i in range(n_modos):
        ax = axs_list[i]

        # ---------------------------------------------------------
        # Ajuste robusto de longitudes:
        # - si niveles tiene un punto extra (base), agregar 0 al modo
        # - si niveles tiene misma longitud, usar modo tal cual
        # ---------------------------------------------------------
        if len(niveles) == n_gdl + 1:
            modo = np.r_[0.0, Vplot[:, i]]
            y = niveles
        elif len(niveles) == n_gdl:
            modo = Vplot[:, i]
            y = niveles
        else:
            raise ValueError(
                f"plot_modes_grid: dimensiones incompatibles. "
                f"Vn tiene {n_gdl} filas y niveles tiene {len(niveles)} valores."
            )

        ax.plot(modo,  y, "-o", color=COLOR_MODO, lw=1.05, ms=2.5)
        ax.plot(-modo, y, "--o", color=COLOR_INV,  lw=0.90, ms=2.2, alpha=0.95)
        ax.plot(np.zeros_like(y), y, "-", color=COLOR_STRUCT, lw=0.75, alpha=0.85)

        ax.set_title(
            f"{mode_lbl} {i+1}\nT={T[i]:.3f} s",
            color=COLOR_TEXT, fontsize=fs_title, path_effects=halo2
        )
        ax.tick_params(colors=COLOR_TEXT, labelsize=8)
        ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-1.1, 1.1)

        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["-1", "0", "1"], color=COLOR_TEXT, fontsize=8)

    for j in range(n_modos, len(axs_list)):
        axs_list[j].axis("off")

    ylbl = tr("b5_height_lbl")
    for r in range(nrows):
        axs_list[r * ncols].set_ylabel(ylbl, color=COLOR_TEXT)

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

    fig, ax = plt.subplots(figsize=(5.1, 4.8))
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
    ax.text(
        x_center+base_half+0.06, 0,
        tr("b5_fixed_base_lbl"),
        va="center", fontsize=fs, color=COLOR_BASE, path_effects=HALO
    )

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

    ax.set_title(tr("b5_fixed_model_lbl"), color=COLOR_TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel(tr("b5_height_lbl"), color=COLOR_TEXT)
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

    fig, ax = plt.subplots(figsize=(5.1, 4.8))
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
    xk_txt  = x_center - 0.12
    xm      = x_center
    xm_txt  = x_center + 0.10

    XMAX = max(abs(xk_txt) + 0.45, abs(xm_txt) + 0.65)

    # ✅ ÚNICA línea azul del edificio: desde -1 hasta el techo
    ax.plot([x_center, x_center], [-1.0, y_top], color=COLOR_SPR, lw=lw_ed)

    # ✅ Base fija corta en y=-1
    base_half = 0.12
    ax.plot([x_center-base_half, x_center+base_half], [-1.0, -1.0],
            color=COLOR_BASE, lw=5, solid_capstyle="round")
    ax.text(
        x_center+base_half+0.06, -1.0,
        tr("b5_fixed_base_lbl"),
        va="center", fontsize=fs, color=COLOR_BASE, path_effects=HALO
    )

    # ----------------- Textos K (a la izquierda del eje) -----------------
    ax.text(
        xk_txt, -0.5,
        f"$K_{{ais}}={k_iso_plot:.1f}$ Tf/m",
        color=COLOR_LABEL_SPR, fontsize=fs,
        va="center", ha="right", path_effects=HALO
    )

    for i in range(2, len(yline)):
        ymid = 0.5 * (yline[i-1] + yline[i])
        k_i = float(k_story_ais_plot[i-2])
        ax.text(
            xk_txt, ymid,
            f"$K_{{{i-1}}}={k_i:.1f}$ Tf/m",
            color=COLOR_LABEL_SPR, fontsize=fs,
            va="center", ha="right", path_effects=HALO
        )

    # ----------------- Masas -----------------
    ax.plot(xm, 0.0, "o", color=COLOR_MASS, markersize=ms)
    ax.text(
        xm_txt, 0.0,
        f"$M_{{0}}={M_cond_ais[0,0]:.3f}$ Tf·s²/m",
        color=COLOR_LABEL_MASS, fontsize=fs,
        va="center", ha="left", path_effects=HALO
    )

    for i, yv in enumerate(pisos_y, start=1):
        ax.plot(xm, float(yv), "o", color=COLOR_MASS, markersize=ms)
        ax.text(
            xm_txt, float(yv),
            f"$M_{{{i}}}={M_cond_ais[i,i]:.3f}$ Tf·s²/m",
            color=COLOR_LABEL_MASS, fontsize=fs,
            va="center", ha="left", path_effects=HALO
        )

    # ----------------- Estilo ejes -----------------
    ax.set_title(tr("b5_iso_model_lbl"), color=COLOR_TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel(tr("b5_height_lbl"), color=COLOR_TEXT)

    ax.set_xlim(-XMAX, XMAX)
    ax.set_ylim(-1.5, y_top + 1.0)
    ax.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.35)
    ax.tick_params(colors=COLOR_TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])

    fig.tight_layout()
    _freeze_axes_limits(fig)
    return fig

# =============================================================================
# === BLOQUE 6 (FUNCIONES): UTILIDADES THA / NEWMARK / RAYLEIGH / EXCEL ========
# =============================================================================
import numpy as np
import pandas as pd
import io
import hashlib

def rayleigh_from_w(w_in, zeta):
    w_in = np.asarray(w_in, dtype=float).ravel()
    w_in = w_in[np.isfinite(w_in)]
    zeta = float(zeta)

    if len(w_in) == 0:
        raise ValueError("No hay frecuencias válidas para calcular amortiguamiento.")

    # Caso de 1 solo modo válido
    if len(w_in) == 1:
        w1 = float(w_in[0])
        alpha = 2.0 * zeta * w1
        beta = 0.0
        return float(alpha), float(beta)

    # Caso clásico Rayleigh con 2 modos
    w1, w2 = float(w_in[0]), float(w_in[1])

    A = np.array([
        [1.0 / (2.0 * w1), w1 / 2.0],
        [1.0 / (2.0 * w2), w2 / 2.0]
    ], dtype=float)

    b = np.array([zeta, zeta], dtype=float)
    alpha, beta = np.linalg.solve(A, b)

    return float(alpha), float(beta)

def pick_two_w(w, wmin=1e-6):
    w = np.asarray(w, dtype=float).ravel()
    w = w[np.isfinite(w)]
    w = np.sort(w[w > wmin])

    if len(w) == 0:
        raise ValueError("No hay frecuencias válidas para amortiguamiento (w > wmin).")

    if len(w) == 1:
        return w[:1]

    return w[:2]
    
def ensure_2d(u, v, a):
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    a = np.asarray(a, float)
    if u.ndim == 1:
        u = u[np.newaxis, :]
        v = v[np.newaxis, :]
        a = a[np.newaxis, :]
    return u, v, a

def modal_w(K, M, n_modes=None):
    try:
        from scipy.linalg import eigh

        K = np.asarray(K, float)
        M = np.asarray(M, float)

        # ✅ fuerza simetría numérica (CRÍTICO)
        K = 0.5 * (K + K.T)
        M = 0.5 * (M + M.T)

        w2, _ = eigh(K, M)  # K φ = w² M φ
        w2 = np.real(w2)

    except Exception:
        A = np.linalg.solve(M, K)
        w2, _ = np.linalg.eig(A)
        w2 = np.real(w2)

    w2 = np.sort(np.maximum(w2, 0.0))
    if n_modes is not None:
        w2 = w2[:n_modes]
    return np.sqrt(w2)

def _sig(*arrays, extra=None):
    h = hashlib.sha1()
    for x in arrays:
        a = np.asarray(x)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    if extra is not None:
        h.update(str(extra).encode())
    return h.hexdigest()

def make_excel_per_floor(t, u, v, a, sheet_names):
    t = np.asarray(t, float).ravel()
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    a = np.asarray(a, float)
    if u.ndim != 2 or v.ndim != 2 or a.ndim != 2:
        raise ValueError("u, v, a deben ser matrices 2D (n_dof, n_t).")
    if not (u.shape == v.shape == a.shape):
        raise ValueError("u, v, a deben tener la misma forma.")
    if u.shape[1] != t.shape[0]:
        raise ValueError("t y series no tienen la misma longitud.")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        n = u.shape[0]
        for i in range(n):
            sh = sheet_names[i] if i < len(sheet_names) else f"Level_{i}"
            sh = sh[:31]
            df = pd.DataFrame({
                "t (s)": t,
                "a (m/s²)": a[i, :],
                "v (m/s)": v[i, :],
                "u (m)": u[i, :],
            })
            df.to_excel(writer, sheet_name=sh, index=False)
    output.seek(0)
    return output.getvalue()

