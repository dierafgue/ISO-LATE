# 💻 ISO-LATE  
### *Herramienta Interactiva para Análisis Estructural – Base Fija vs Aislamiento Sísmico*

<p align="center">
  <img src="assets/logo.png" alt="Logo ISO-LATE" width="220"/>
</p>

<p align="center">
  <b>ISO-LATE</b> es una aplicación de ingeniería interactiva desarrollada para <b>simular, analizar y comparar la respuesta sísmica de estructuras 2D</b> con <b>base fija</b> y <b>sistemas con aislamiento sísmico en la base</b>, apoyando procesos de aprendizaje, investigación y diseño preliminar en ingeniería sismorresistente.
</p>

<p align="center">
  🌐 <a href="https://iso-late.streamlit.app" target="_blank"><b>Aplicación en Línea</b></a> • 📘 <a href="#manual-de-usuario">Manual de Usuario</a> • 🧠 <a href="#fundamento-teorico">Fundamento Teórico</a> • ⚙️ <a href="#instalacion">Instalación</a>
</p>

---

## 📌 Tabla de Contenidos
- [Descripción General](#descripcion-general)
- [Características Principales](#caracteristicas-principales)
- [Alcance de Ingeniería](#alcance-de-ingenieria)
- [Fundamento Teórico](#fundamento-teorico)
- [Estructura de la Aplicación](#estructura-de-la-aplicacion)
- [Instalación](#instalacion)
- [Uso](#uso)
- [Manual de Usuario](#manual-de-usuario)
- [Validación y Limitaciones](#validacion-y-limitaciones)
- [Tecnologías Utilizadas](#tecnologias-utilizadas)
- [Autor y Contexto Académico](#autor-y-contexto-academico)
- [Licencia](#licencia)

---

## 🧭 Descripción General

**ISO-LATE** es una herramienta educativa y orientada a la ingeniería que permite a los usuarios:

- Modelar **estructuras aporticadas 2D de múltiples niveles**
- Realizar **análisis dinámico lineal**
- Comparar el comportamiento estructural entre sistemas de **base fija** y **base aislada**
- Visualizar **métricas de respuesta sísmica** de forma clara e intuitiva

La aplicación está especialmente dirigida a:

- Estudiantes de ingeniería estructural  
- Investigadores en ingeniería sísmica  
- Ingenieros en etapa conceptual o de validación preliminar  

---

## ✨ Características Principales

✔️ Definición paramétrica de estructuras 2D (pisos, vanos, geometría)  
✔️ Generación automática de matrices de masa y rigidez  
✔️ Análisis modal y análisis por espectro de respuesta  
✔️ Análisis en el dominio del tiempo mediante el método **Newmark-β**  
✔️ Modelado de aislamiento sísmico (LRB / NRB – equivalente lineal)  
✔️ Comparación lado a lado: **Base Fija vs Base Aislada**  
✔️ Gráficos técnicos limpios y escalables  
✔️ Interfaz interactiva basada en la web (Streamlit)  

---

## 🏗️ Alcance de Ingeniería

La aplicación se enfoca en:

- Comportamiento **elástico lineal**
- Estructuras planas (2D)
- Idealización tipo **shear-building**
- Modelado equivalente lineal para aisladores sísmicos
- Uso educativo y comparativo (no destinado a diseño estructural final)

> ⚠️ **ISO-LATE no está diseñado para reemplazar software profesional de análisis no lineal avanzado** como OpenSees, ETABS o SAP2000.

---

## 📐 Fundamento Teórico

La formulación principal se basa en:

- Análisis matricial de estructuras
- Ecuación de movimiento para sistemas de múltiples grados de libertad (MDOF):

\[
\mathbf{M}\ddot{\mathbf{u}} + \mathbf{C}\dot{\mathbf{u}} + \mathbf{K}\mathbf{u} = -\mathbf{M}\mathbf{r} \ddot{u}_g
\]

donde:

- **M** = matriz de masa  
- **C** = matriz de amortiguamiento  
- **K** = matriz de rigidez  
- **u** = vector de desplazamientos relativos  
- **ug** = aceleración del terreno  

Además, se aplican:

- Superposición modal  
- Amortiguamiento de Rayleigh  
- Integración numérica mediante el método **Newmark-β**  
- Teoría simplificada equivalente lineal para aislamiento sísmico  

### Referencias Técnicas

- Chopra, A.K. – *Dynamics of Structures*  
- ASCE 7 / ASCE 41  
- FEMA 440 / FEMA P-1050  

---

## 🧩 Estructura de la Aplicación

```text
ISO-LATE/
│
├── app.py                     # Aplicación principal en Streamlit
├── funciones_usuario.py       # Funciones de modelado estructural y análisis dinámico
├── requirements.txt           # Dependencias de Python
├── .streamlit/
│   └── config.toml            # Configuración visual y del servidor
├── assets/                    # Imágenes, logotipos e íconos
│   └── logo.png
├── data/                      # Registros sísmicos (opcional)
└── README.md                  # Documentación del proyecto
