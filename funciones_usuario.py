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

def graficar_respuesta_por_piso(
    t: np.ndarray,
    u_t: np.ndarray,
    v_t: np.ndarray,
    a_t: np.ndarray,
    alturas: Union[np.ndarray, list],
    t_total: float,
    *args, **kwargs  # ← 🔹 Permite argumentos adicionales como nombre_piso sin error
):
    """
    Grafica para cada piso: desplazamiento, velocidad y aceleración.
    Auto-adapta a Streamlit si está disponible (st.pyplot); si no, usa plt.show().
    """
    nombre_piso = kwargs.get("nombre_piso", None)  # ← 🔹 Si no se pasa, no falla

    # Intento de import local (no obliga a dependencia fuerte con Streamlit)
    _HAS_ST = False
    try:
        import streamlit as st  # type: ignore
        _HAS_ST = True
    except Exception:
        pass

    # 🎨 Paleta Arctic Dark Pastel refinada
    COLOR_BG    = '#2B3141'
    COLOR_TEXT  = '#E8EDF2'
    COLOR_GRID  = '#5B657A'
    COLOR_LINE  = ['#7EB6FF', '#FFD180', '#A8E6CF']
    COLOR_MAX   = '#FFB3B3'
    COLOR_MIN   = '#89D6FF'
    LEG_FACE    = '#363C4A'
    LEG_EDGE    = '#A7B1C5'

    t = np.asarray(t, dtype=float).ravel()
    u_t = np.asarray(u_t, dtype=float)
    v_t = np.asarray(v_t, dtype=float)
    a_t = np.asarray(a_t, dtype=float)
    n_pisos = u_t.shape[0]

    for i in range(n_pisos):
        fig, axs = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
        fig.patch.set_facecolor(COLOR_BG)

        respuestas = [u_t, v_t, a_t]
        etiquetas = ["Desplazamiento [m]", "Velocidad [m/s]", "Aceleración [m/s²]"]

        for j in range(3):
            serie = respuestas[j][i]
            ax = axs[j]
            ax.set_facecolor(COLOR_BG)

            ax.plot(t, serie, color=COLOR_LINE[j], linewidth=1.0, alpha=0.95)

            # --- Máximo y mínimo ---
            max_idx = int(np.argmax(serie))
            min_idx = int(np.argmin(serie))
            max_t, max_val = t[max_idx], serie[max_idx]
            min_t, min_val = t[min_idx], serie[min_idx]
            ax.plot(max_t, max_val, 'o', color=COLOR_MAX, markersize=4.5)
            ax.plot(min_t, min_val, 's', color=COLOR_MIN, markersize=4.5)

            legend_elements = [
                Line2D([0], [0], marker='o', color='w',
                       label=f'Max: {max_val:.4e} @ {max_t:.2f}s',
                       markerfacecolor=COLOR_MAX, markersize=6),
                Line2D([0], [0], marker='s', color='w',
                       label=f'Min: {min_val:.4e} @ {min_t:.2f}s',
                       markerfacecolor=COLOR_MIN, markersize=6)
            ]
            leg = ax.legend(handles=legend_elements, loc="upper right",
                            facecolor=LEG_FACE, edgecolor=LEG_EDGE,
                            framealpha=0.9, fontsize=8.5)
            for t_ in leg.get_texts():
                t_.set_color(COLOR_TEXT)

            ax.set_ylabel(etiquetas[j], color=COLOR_TEXT, fontsize=9)
            ax.grid(True, color=COLOR_GRID, linestyle=':', alpha=0.35)
            ax.tick_params(colors=COLOR_TEXT, labelsize=8)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color(COLOR_GRID)
            ax.spines['left'].set_color(COLOR_GRID)

        axs[-1].set_xlabel("Tiempo [s]", color=COLOR_TEXT, fontsize=9)
        axs[-1].set_xlim(0, float(t_total))

        # --- Título adaptativo ---
        titulo = f"Respuestas del piso {nombre_piso}" if nombre_piso else f"Respuestas del piso {i + 1}"
        fig.suptitle(titulo, fontsize=12, color=COLOR_TEXT, fontweight='semibold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # --- Mostrar ---
        if _HAS_ST:
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
                                   peso_especifico: float = 2.4,
                                   sobrecarga_muerta: float = 0.0) -> np.ndarray:
    """
    Calcula la matriz diagonal de masas lumped por piso (un grado de libertad por piso).
    Las masas están en unidades [Tf·s²/m], consistentes con rigidez en [Tf/m].
    """
    g = 9.81  # m/s²

    # Alturas únicas mayores que 0
    alturas = sorted({round(y, 6) for (x, y, _) in nodes if y > 0})
    n_pisos = len(alturas)
    masas_por_piso = np.zeros(n_pisos, dtype=float)

    # Mapa id → coordenadas
    node_by_id = {nid: (x, y, nid) for (x, y, nid) in nodes}

    for n1, n2, tipo in element_node_pairs:
        x1, y1, _ = node_by_id[n1]
        x2, y2, _ = node_by_id[n2]
        L = float(np.hypot(x2 - x1, y2 - y1))
        A = float(propiedades[tipo]["A"])
        peso_elem = peso_especifico * A * L  # [Tf]

        if tipo == "col":
            # Peso tributario mitad abajo y mitad arriba
            altura_inf = min(y1, y2)
            altura_sup = max(y1, y2)
            if altura_inf > 0:
                idx_inf = alturas.index(round(altura_inf, 6))
                masas_por_piso[idx_inf] += (peso_elem / 2) / g
            if altura_sup > 0:
                idx_sup = alturas.index(round(altura_sup, 6))
                masas_por_piso[idx_sup] += (peso_elem / 2) / g

        elif tipo == "viga":
            # Viga pertenece completamente al nivel donde está
            altura_viga = round(y1, 6)
            if altura_viga in alturas:
                idx = alturas.index(altura_viga)
                masas_por_piso[idx] += (peso_elem / g)
                if sobrecarga_muerta > 0:
                    peso_sob = sobrecarga_muerta * L
                    masas_por_piso[idx] += (peso_sob / g)

    # --- Resultado final ---
    return np.diag(masas_por_piso)

def plot_structure(nodes, elements, nodos_restringidos,
                   gdl_dinamicos_local=None, gdl_estaticos_local=None,
                   gdl_map=None, propiedades=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import matplotlib.transforms as mtrans

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

    # ---- OpenSees + opsvis (solo geometría) ----
    try:
        import openseespy.opensees as ops
        import opsvis as opsv
    except Exception as e:
        raise RuntimeError(f"Falta OpenSeesPy/opsvis: {e}")

    # mapeo coord->nid (robusto)
    coord_to_nid = {(round(_to_float(x), 8), round(_to_float(y), 8)): _to_int(nid)
                    for (x, y, nid) in nodes}

    def nid_from_xy(x, y):
        key = (round(_to_float(x), 8), round(_to_float(y), 8))
        if key in coord_to_nid:
            return coord_to_nid[key]
        d = (xs - _to_float(x))**2 + (ys - _to_float(y))**2
        return _to_int(nodes[int(np.argmin(d))][2])

    # ---- crear modelo temporal en OpenSees ----
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    for (x, y, nid) in nodes:
        ops.node(_to_int(nid), _to_float(x), _to_float(y))

    if nodos_restringidos:
        for nid in nodos_restringidos:
            ops.fix(_to_int(nid), 1, 1, 1)

    ops.geomTransf("Linear", 1)

    # propiedades dummy solo para dibujar
    A, E, Iz = 1.0, 1.0, 1.0

    etag = 1
    for el in elements:
        ni = nid_from_xy(el.node_start[0], el.node_start[1])
        nj = nid_from_xy(el.node_end[0],   el.node_end[1])
        if int(ni) == int(nj):
            continue
        ops.element("elasticBeamColumn", etag, int(ni), int(nj), A, E, Iz, 1)
        etag += 1

    # ---- dibujo crudo con opsvis (ESCALAS BUENAS) ----
    opsv.plot_model(node_labels=False, element_labels=False, ax=ax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, alpha=0.35)

    # ==========================
    # ✅ SOLO RETOQUE: colores + “ENANOS” para edificios altos
    #    - líneas (elementos)
    #    - nodos (collections)
    #    - apoyos/empotramientos (patches)  <-- AQUÍ estaba tu problema
    # ==========================

    # factor global (30 pisos -> ~0.30-0.35)
    scale = float(np.clip(1.00 - 0.023 * n_pisos, 0.30, 1.00))

    # --- 1) elementos ---
    lw_elem = float(np.clip((1.35 - 0.02 * n_pisos) * scale, 0.55, 1.25))
    for ln in ax.lines:
        ln.set_color(ELEM_C)
        ln.set_linewidth(lw_elem)

        # si hay markers en líneas (a veces opsvis), baja también
        try:
            ms = ln.get_markersize()
            if ms is not None and ms > 0:
                ln.set_markersize(max(1.5, ms * scale))
        except Exception:
            pass

    # --- 2) nodos (scatter collections) ---
    # sizes son en pt^2 -> bajar agresivo en 30 pisos
    base_node = float(np.clip(14.0 - 0.38 * n_pisos, 2.6, 10.0))   # 30 -> 2.6 (clamp)
    node_size = base_node * (scale**2)
    sup_size  = node_size * 1.05

    for col in ax.collections:
        try:
            offs = col.get_offsets()
            if offs is None or len(offs) == 0:
                continue
            yoff = np.asarray(offs)[:, 1]
            is_base = np.isclose(yoff, y_min, atol=1e-6)

            col.set_sizes(np.where(is_base, sup_size, node_size).astype(float))
            col.set_facecolor(ELEM_C)
            col.set_edgecolor(TEXT)
            col.set_linewidths(float(np.clip(0.45 * scale, 0.18, 0.45)))
            col.set_alpha(1.0)
        except Exception:
            pass

    # --- 3) empotramientos/soportes (patches) ---
    # estos son los cuadrados grandes: hay que escalarlos aquí sí o sí
    # criterio: patches cerca de la base
    base_band = 0.10 * max(Ly, 1.0)  # banda “base”
    for p in ax.patches:
        try:
            bb = p.get_extents()
            cy = 0.5 * (bb.y0 + bb.y1)
            if abs(cy - y_min) <= base_band:
                # escalar alrededor del centro del patch
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
    # NUMERACIÓN ULTRA COMPACTA (30 pisos)
    # ==========================
    if n_pisos <= 10:
        font_node = 8
        step_i = 1
        weight = "bold"
    elif n_pisos <= 18:
        font_node = 7
        step_i = 2
        weight = "bold"
    elif n_pisos <= 26:
        font_node = 6
        step_i = 3
        weight = "bold"
    else:
        font_node = 4          # 🔥 más pequeño aún
        step_i = 5             # 🔥 menos etiquetas
        weight = "normal"      # 🔥 sin negrita

    # offsets: un poco más lejos del nodo
    dx = 0.020 * max(Lx, 1.0)
    dy = float(np.clip(0.018 * (Ly / n_pisos), 0.08, 0.22))

    y_levels_r = [round(v, 8) for v in y_levels]
    last_level = len(y_levels_r) - 1

    for (x, y, nid) in nodes:
        x = _to_float(x)
        y = _to_float(y)
        nid = _to_int(nid)
        yl = round(y, 8)

        try:
            i_level = y_levels_r.index(yl)
        except ValueError:
            i_level = 0

        # solo cada step y siempre el último
        if (i_level % step_i) != 0 and i_level != last_level:
            continue

        ax.text(
            x + dx, y + dy, str(nid + label_add),
            fontsize=font_node,
            color=NODE_TXT,
            fontweight=weight,
            ha="left", va="center",
            path_effects=HALO,
            zorder=50
        )

    # ==========================
    # ✅ LÍMITES CUADRADOS (como tu FIG1)
    # ==========================
    span = max(Lx, Ly)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    margin = 0.12 * span
    ax.set_xlim(cx - 0.5 * span - margin, cx + 0.5 * span + margin)
    ax.set_ylim(cy - 0.5 * span - margin, cy + 0.5 * span + margin)

    # ==========================
    # ✅ Texto “crudo” (pequeño, no gigante)
    # ==========================
    ax.set_title(f"Pórtico 2D – {n_pisos} pisos, {n_vanos} vanos",
                 fontsize=11, color=TEXT, pad=6)
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
    return a_rel + (ag_g * 9.81)[np.newaxis, :]


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
    ag = ag_g * 9.81
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
    """Convierte aceleraciones relativas a absolutas (a_abs = a_rel + ag*9.81)."""
    return a_rel + (ag_g * 9.81)[np.newaxis, :]

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
def calcular_matriz_masas_con_aislador(nodes, element_node_pairs, propiedades,
                                       peso_especifico=2.4, sobrecarga_muerta=0.0):
    """
    Calcula la matriz de masas condensada (diagonal) de un modelo
    con aislador en la base, asignando un DOF adicional (aislador).
    """
    g = 9.81
    alturas = sorted(set([y for (x, y, _) in nodes if y > 0]))
    n_pisos = len(alturas)
    masas_por_piso = np.zeros(n_pisos + 1)

    for piso_idx, altura in enumerate(alturas):
        for n1, n2, tipo in element_node_pairs:
            xi, yi, _ = nodes[n1]
            xj, yj, _ = nodes[n2]
            L = np.hypot(xj - xi, yj - yi)
            A = propiedades[tipo]['A']

            if tipo == 'col':
                altura_inf = min(yi, yj)
                altura_sup = max(yi, yj)
                if abs(altura_sup - altura) < 1e-6:
                    peso = peso_especifico * A * L
                    masas_por_piso[piso_idx + 1] += peso / g
            elif tipo == 'viga':
                if abs(yi - altura) < 1e-6 and abs(yj - altura) < 1e-6:
                    peso = peso_especifico * A * L
                    masas_por_piso[piso_idx + 1] += peso / g
                    if sobrecarga_muerta > 0:
                        peso_sob = sobrecarga_muerta * L
                        masas_por_piso[piso_idx + 1] += peso_sob / g

    masas_por_piso[0] = (2 / 3) * masas_por_piso[1]
    return np.diag(masas_por_piso)


# =============================================================================
# === CLASE ELEMENTO REDUCIDO LIBRE (2 DOF POR NODO: vx, θ) ===================
# =============================================================================
class ElementReducidoLibre:
    """
    Elemento viga-columna 2D reducido con grados de libertad [vx, θ] en cada nodo.
    Ideal para modelos condensados o sistemas base-aislados 2D.
    """
    def __init__(self, node_start, node_end, E, I, A, gdl_map):
        self.node_start = node_start
        self.node_end = node_end
        self.E = E
        self.I = I
        self.A = A
        self.length = np.hypot(node_end[0] - node_start[0],
                               node_end[1] - node_start[1])

        self.k_local = beam_stiffness_2D(E, I, A, self.length)
        self.T = transformation_matrix(node_start, node_end)
        self.k_global = self.T.T @ self.k_local @ self.T
        self.dofs = self.assign_dofs(gdl_map)

    def assign_dofs(self, gdl_map):
        """Asigna los GDL globales vx y θ (elimina vy)."""
        i = self.node_start[2]
        j = self.node_end[2]
        dofs_i = [gdl_map.get((i, 'vx')), None, gdl_map.get((i, 'theta'))]
        dofs_j = [gdl_map.get((j, 'vx')), None, gdl_map.get((j, 'theta'))]
        return dofs_i + dofs_j
    
# =============================================================================
# === FUNCIÓN PARA GENERAR MAPA DE GDL REDUCIDO LIBRE =========================
# =============================================================================
def generar_gdl_map_reducido_libre(nodes):
    """Genera el mapa de GDL global para un modelo 2D reducido (vx, θ)."""
    niveles = sorted(set([y for _, y, _ in nodes]))
    gdl_vx_por_nivel = {nivel: i for i, nivel in enumerate(niveles)}
    gdl_counter = len(niveles)
    gdl_map = {}

    for (x, y, nid) in nodes:
        gdl_map[(nid, 'vx')] = gdl_vx_por_nivel[y]
        gdl_map[(nid, 'vy')] = None
        gdl_map[(nid, 'theta')] = gdl_counter
        gdl_counter += 1

    return gdl_map, gdl_counter


# =============================================================================
# === FUNCIÓN: ENSAMBLAR MATRIZ GLOBAL REDUCIDA LIBRE =========================
# =============================================================================
def assemble_global_stiffness_reducido_libre(elements, total_dofs):
    """Ensamblaje global (solo DOF vx y θ) para elementos reducidos libres."""
    K = np.zeros((total_dofs, total_dofs))
    idx_local = [0, 2, 3, 5]

    for el in elements:
        k_reducido = el.k_global[np.ix_(idx_local, idx_local)]
        dofs = el.dofs
        for m in range(4):
            for n in range(4):
                dof_m = dofs[idx_local[m]]
                dof_n = dofs[idx_local[n]]
                if dof_m is not None and dof_n is not None:
                    K[dof_m, dof_n] += k_reducido[m, n]
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
# === FUNCIÓN: Diseño de aisladores LRB (Automático o por Período Objetivo) ===
# =============================================================================
def diseno_aislador_LRB(
    SD1, SDS, T_sin, Mc, nodos_restringidos,
    modo_automatico=True, modo_periodo_objetivo=False, T_objetivo=None
):
    """
    Diseño de aisladores LRB (2D) sin restricciones geométricas impuestas.
    Usa la masa TOTAL del sistema FIJO (Mc). Unidades: Tonf, Tonf/m, m.
    """
    import math
    import numpy as np

    # -------------------- DATOS BASE --------------------
    SM1 = 1.5 * SD1
    T_usar = float(np.atleast_1d(T_sin)[0])  # T1 sin aislador
    g = 9.81

    # ---------------- MASA / PESO DEL SISTEMA FIJO ----------------
    if Mc is None or np.ndim(Mc) != 2 or Mc.shape[0] != Mc.shape[1]:
        raise ValueError("Mc debe ser una matriz cuadrada de masas (sistema FIJO).")

    M_super = float(np.sum(np.diag(Mc)))  # [Tonf·s²/m]
    W_total = M_super * g                 # [Tonf]
    n_ais = max(int(len(nodos_restringidos)), 1)
    W_individual = W_total / n_ais

    # ---------------- PROPIEDADES MATERIALES ----------------
    Kver_ais = 1805                                   # [kN/m] (no usado)
    Kver_ais_tonfm = Kver_ais * 0.101972 * 1000       # [Tonf/m]
    sigma_L = 10                                      # [MPa] (plomo)
    sigma_L_tonf = sigma_L * 0.001 * 0.101972 * (1000**2)  # [Tonf/m²]
    G = 0.45                                          # [MPa] (caucho)
    G_tonf = G * 0.001 * 0.101972 * (1000**2)              # [Tonf/m²]
    delta_L = 0.025                                   # [m] espesor de capa de goma

    # Tabla empírica B_M(β)
    betta_vals = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    bd_vals    = [0.8,  1.0,  1.2,  1.5,  1.7,  1.9,  2.0]

    # Variables que devolveremos
    Q_d_total_LB = None
    K_D_total_LB = None
    k_M = None
    betta_M = None
    D_L = None
    D_B = None
    T_r = None
    D_M = None
    T_final = None
    iteraciones = 0

    # ===================== MODO AUTOMÁTICO =====================
    if modo_automatico and not modo_periodo_objetivo:
        print("\n⚙️  MODO: Diseño Automático (iterativo sin límites)\n")

        # Núcleo de plomo (por carga vertical ~5% W_total / n_ais)
        D_L = math.sqrt((0.05 * W_total * 4.0) / (n_ais * math.pi * sigma_L_tonf))

        # Relaciones base (sin restricciones geométricas)
        D_B = 4.0 * D_L
        T_r = D_L

        # Inicialización de desplazamiento (compatibilidad espectral)
        D_M = (SDS * g) / ((2.0 * math.pi**2) / T_usar)
        tol, max_iter = 1e-5, 100
        error, iteraciones = 1.0, 0

        while error > tol and iteraciones < max_iter:
            # Rigidez de corte total de gomas
            K_D_total = n_ais * (G_tonf * math.pi * (D_B**2 - D_L**2)) / (4.0 * T_r)
            K_D_total_LB = 0.85 * K_D_total

            # Fuerza de fluencia total del plomo
            Q_d_total = n_ais * (math.pi * D_L**2) * sigma_L_tonf / 4.0
            Q_d_total_LB = 0.85 * Q_d_total

            # Rigidez efectiva total del sistema aislado
            k_M = K_D_total_LB + Q_d_total_LB / D_M

            # Período efectivo
            T_M = 2.0 * math.pi * math.sqrt(W_total / (k_M * g))

            # Amortiguamiento equivalente por histéresis
            betta_M = (2.0 * Q_d_total_LB * (D_M - delta_L)) / (math.pi * k_M * D_M**2)
            betta_M = max(betta_M, 0.0)

            # Reducción espectral por β (interpolada)
            if betta_M <= 0.02:
                B_M = 0.8
            elif betta_M >= 0.50:
                B_M = 2.0
            else:
                B_M = float(np.interp(betta_M, betta_vals, bd_vals))

            # Nueva demanda de desplazamiento
            D_M_new = (g * SM1 * T_M) / (4.0 * math.pi**2 * B_M)

            error = abs(D_M_new - D_M)
            D_M = D_M_new
            iteraciones += 1

        T_final = T_M

    # ================= MODO PERÍODO OBJETIVO ==================
    elif modo_periodo_objetivo and not modo_automatico:
        print("\n🎯  MODO: Diseño por Período Objetivo (ajuste geométrico automático)\n")

        if T_objetivo is None or T_objetivo <= 0:
            raise ValueError("Debe indicar T_objetivo > 0 cuando modo_periodo_objetivo=True.")

        # Geometría base (misma lógica que el modo automático)
        D_L = math.sqrt((0.05 * W_total * 4.0) / (n_ais * math.pi * sigma_L_tonf))

        # Configuración de ajuste automático
        rel_DB_ini   = 4.0   # D_B/D_L inicial
        rel_DB_min   = 2.5   # límite inferior razonable para D_B/D_L
        margen_Tr    = 1.10  # 10% por encima del mínimo para asegurar k_M > K_D_LB

        # Rigidez requerida por el período objetivo
        k_M = (4.0 * math.pi**2 * W_total) / (g * T_objetivo**2)

        # Búsqueda: primero solo ajusta T_r; si no alcanza, reduce D_B/D_L gradualmente
        solucion = False
        for rel_DB in np.linspace(rel_DB_ini, rel_DB_min, num=8):
            D_B = rel_DB * D_L

            # T_r mínimo para cumplir k_M > K_D_total_LB:
            # K_D_total_LB = 0.85 * n_ais * (G*pi*(D_B^2 - D_L^2)) / (4*T_r)  <  k_M
            #  => T_r_min = 0.85 * n_ais * (G*pi*(D_B^2 - D_L^2)) / (4*k_M)
            Tr_min = (0.85 * n_ais * (G_tonf * math.pi * (D_B**2 - D_L**2))) / (4.0 * k_M)

            # Impón T_r >= D_L (práctica usual) y aplica margen
            T_r = max(D_L, margen_Tr * Tr_min)

            # Recalcular K_D_total con este T_r
            K_D_total = n_ais * (G_tonf * math.pi * (D_B**2 - D_L**2)) / (4.0 * T_r)
            K_D_total_LB = 0.85 * K_D_total

            if k_M > K_D_total_LB:
                solucion = True
                break

        if not solucion:
            # Sin solución razonable: calcula un T_objetivo_min de referencia para tu geometría más blanda (rel_DB_min, T_r=D_L)
            D_B_tmp = rel_DB_min * D_L
            K_D_tmp = n_ais * (G_tonf * math.pi * (D_B_tmp**2 - D_L**2)) / (4.0 * D_L)
            K_D_tmp_LB = 0.85 * K_D_tmp
            # k_M debe ser > K_D_tmp_LB -> T_min = 2π * sqrt(W_total / (g * k_M_req)) con k_M_req = K_D_tmp_LB*(1+margen)
            k_req = K_D_tmp_LB * 1.02
            T_min_posible = 2.0 * math.pi * math.sqrt(W_total / (g * k_req))
            raise ValueError(
                ("Para T_objetivo = {:.3f}s no se logró k_M > K_D_total_LB aun ajustando T_r y D_B/D_L. "
                 "Pruebe con T_objetivo ≥ {:.3f}s, reduzca G o aumente n_ais.")
                .format(T_objetivo, T_min_posible)
            )

        # Con geometría ajustada, calcula fuerzas y desplazamientos
        Q_d_total = n_ais * (math.pi * D_L**2) * sigma_L_tonf / 4.0
        Q_d_total_LB = 0.85 * Q_d_total

        # Desplazamiento compatible con la rama equivalente
        D_M = Q_d_total_LB / (k_M - K_D_total_LB)

        # Amortiguamiento equivalente
        betta_M = (2.0 * Q_d_total_LB * (D_M - delta_L)) / (math.pi * k_M * D_M**2)
        betta_M = max(betta_M, 0.0)

        T_final = float(T_objetivo)
        iteraciones = 0

    else:
        raise ValueError("Activa exactamente uno de los modos: modo_automatico XOR modo_periodo_objetivo.")

    # ---------------- PROPIEDADES POR AISLADOR ----------------
    keff_1ais = (K_D_total_LB / n_ais) + (Q_d_total_LB / D_M) / n_ais          # [Tonf/m]
    c_1ais = betta_M * 2.0 * math.sqrt((keff_1ais * W_individual) / g)          # [Tonf·s/m]
    k_inicial_1ais = (math.pi * D_L**2 * sigma_L_tonf / 4.0) / delta_L          # [Tonf/m]
    yield_1ais =  (math.pi * D_L**2 * sigma_L_tonf / 4.0)                        # [Tonf]
    k_post_1ais = K_D_total_LB / n_ais                                          # [Tonf/m]
    ratio_postfluencia_1ais = ((G_tonf * math.pi * (D_B**2 - D_L**2)) / (4.0 * T_r)) / k_inicial_1ais

    # ---------------- SALIDA EN CONSOLA ----------------
    print("====================================================")
    print(f"D_L = {D_L:.3f} m | D_B = {D_B:.3f} m | T_r = {T_r:.3f} m")
    print(f"T_M = {T_final:.3f} s | D_M = {D_M:.4f} m")
    print(f"k_M = {k_M:.3f} Tonf/m | β_M = {betta_M:.3f}")
    print(f"Keff = {keff_1ais:.3f} Tonf/m | c = {c_1ais:.3f} Tonf·s/m")
    print("====================================================")

    # ---------------- DICCIONARIO DE RESULTADOS ----------------
    return {
        "T_M": T_final,
        "beta_M": betta_M,
        "D_M": D_M,
        "k_M": k_M,
        "keff_1ais": keff_1ais,
        "c_1ais": c_1ais,
        "k_inicial_1ais": k_inicial_1ais,
        "yield_1ais": yield_1ais,
        "k_post_1ais": k_post_1ais,
        "ratio_postfluencia_1ais": ratio_postfluencia_1ais,
        "D_L": D_L,
        "D_B": D_B,
        "T_r": T_r,
        "delta_L": delta_L,
        "iteraciones": iteraciones,
        "Kver_ais_tonfm": Kver_ais_tonfm
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
    titulo='Ciclo Histerético Bilineal – Aislador LRB',
    *, savepath=None, show=True
):
    """
    Dibuja el ciclo bilineal LRB con estética Arctic Dark.
    Todas las unidades en Tonf y metros (Tonf/m, Tonf, m).
    """
    V, F2, Keff_calc, dy = rombo_LRB(Ke, Kp, Fy, dy, D2)
    Keff_usar = Keff_ref if Keff_ref is not None else Keff_calc

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill(V[:,0], V[:,1], color=COLOR_FILL, alpha=0.10)

    # --- Tramos coloreados ---
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

    # --- Línea de rigidez efectiva ---
    ax.plot([0, D2], [0, F2], '--', lw=2.0, color=COLOR_KEFF)

    # --- Asegura transformaciones listas para anotaciones ---
    fig.canvas.draw()

    # --- Textos en segmentos ---
    put_text_on_segment(ax, V[0], V[1], "Ke", color=COLOR_KE)
    put_text_on_segment(ax, V[1], V[2], "Kp", color=COLOR_KP)
    put_text_on_segment(ax, V[2], V[3], "Ke", color=COLOR_KE)
    put_text_on_segment(ax, V[3], V[4], "Kp", color=COLOR_KP)
    put_text_on_segment(ax, V[4], V[5], "Ke", color=COLOR_KE)
    put_text_on_segment(ax, V[5], V[6], "Kp", color=COLOR_KP)
    put_text_on_segment(ax, (0,0), (D2, F2), "Keff", color=COLOR_KEFF)

    # --- Límites automáticos ---
    xmax_abs = np.max(np.abs(V[:,0]))
    ymax_abs = np.max(np.abs(V[:,1]))
    ax.set_xlim(-xmax_abs*1.35, xmax_abs*1.35)
    ax.set_ylim(-ymax_abs*1.30, ymax_abs*1.30)

    # --- Cajas en vértices ---
    box_vertex_outside(ax, V, 1, f"Δ₁ = {dy:.4f} m\nF₁ = {Fy:.3f} Tonf")
    box_vertex_outside(ax, V, 2, f"Δ₂ = {D2:.4f} m\nF₂ = {F2:.3f} Tonf")
    box_vertex_outside(ax, V, 4, f"−Δ₂ = {D2:.4f} m\n−F₂ = {F2:.3f} Tonf")

    # --- Leyenda ---
    handles = [
        Line2D([0],[0], color=COLOR_KE,   lw=2.4, label=f'Ke = {Ke:.3f} Tonf/m'),
        Line2D([0],[0], color=COLOR_KP,   lw=2.4, label=f'Kp = {Kp:.3f} Tonf/m'),
        Line2D([0],[0], color=COLOR_KEFF, lw=2.0, linestyle='--', label=f'Keff = {Keff_usar:.3f} Tonf/m'),
    ]
    leg = ax.legend(handles=handles, loc='upper left', frameon=True, framealpha=0.9)
    for t in leg.get_texts():
        t.set_color(COLOR_TEXT)

    # --- Ejes y formato ---
    ax.axhline(0, color=GRID_ZERO, lw=1.1)
    ax.axvline(0, color=GRID_ZERO, lw=1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(titulo, fontweight='bold', color='#F4F6FA')
    ax.set_xlabel('Desplazamiento Δ (m)')
    ax.set_ylabel('Fuerza Cortante F (Tonf)')
    ax.grid(True, alpha=0.45)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300)
    if show:
        plt.show()
    return fig, ax

# ---------- 5) Wrapper cómodo ----------
def plot_lrb_from_resultados(resultados_ais, *,
                             titulo='Ciclo Histerético Bilineal – Aislador LRB',
                             savepath=None, show=True):
    """
    Usa los resultados del diseño del aislador (en Tonf y m)
    y grafica el ciclo histerético en unidades Tonf/m, Tonf, m.
    """
    Ke   = resultados_ais['k_inicial_1ais']   # Tonf/m
    Kp   = resultados_ais['k_post_1ais']      # Tonf/m
    Fy   = resultados_ais['yield_1ais']       # Tonf
    dy   = resultados_ais['c_1ais']           # m
    D2   = resultados_ais['D_M']              # m
    Keff = resultados_ais['keff_1ais']        # Tonf/m

    return plot_ciclo_histeretico_lrb(
        Ke, Kp, Fy, dy, D2, Keff_ref=Keff,
        titulo=titulo, savepath=savepath, show=show
    )

