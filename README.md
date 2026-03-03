# 💻 ISO-LATE  
### *Herramienta Interactiva para Análisis Estructural – Base Fija vs Aislamiento Sísmico*

<p align="center">
  <img src="assets/logo.png" alt="Logo ISO-LATE" width="400"/>
</p>

<p align="center">
  <b>ISO-LATE</b> es una aplicación de ingeniería interactiva desarrollada para <b>simular, analizar y comparar la respuesta sísmica de estructuras 2D</b> con <b>base fija</b> y <b>sistemas con aislamiento sísmico en la base</b>.
</p>

<p align="center">
  🌐 <a href="https://iso-late.streamlit.app/" target="_blank"><b>Acceder a la Aplicación en Línea</b></a>
</p>

---

## 📌 Tabla de Contenidos
- [Descripción General](#descripcion-general)
- [Características Principales](#caracteristicas-principales)
- [Alcance de Ingeniería](#alcance-de-ingenieria)
- [Fundamento Teórico](#fundamento-teorico)
- [Estructura de la Aplicación](#estructura-de-la-aplicacion)
- [Manual de Usuario](#manual-de-usuario)

---

## 🧭 Descripción General

**ISO-LATE** es una herramienta educativa y orientada a la ingeniería que permite a los usuarios:

- Modelar **estructuras aporticadas 2D de múltiples niveles**
- Realizar **análisis dinámico lineal**
- Comparar el comportamiento estructural entre sistemas de **base fija** y **base aislada**
- Visualizar **métricas de respuesta sísmica** de forma clara e intuitiva
- Entender las diferencias y ventajas del uso de **sistemas de aislamiento**

---

## ✨ Características Principales

✔️ Definición paramétrica de estructuras 2D  
✔️ Generación automática de matrices de masa y rigidez  
✔️ Análisis modal y análisis por espectro de respuesta  
✔️ Análisis en el dominio del tiempo mediante el método **Newmark-β**  
✔️ Modelado de aislamiento sísmico (LRB / NRB – equivalente lineal)  
✔️ Comparación lado a lado: **Base Fija vs Base Aislada**  
✔️ Gráficos técnicos limpios y escalables  
✔️ Interfaz interactiva basada en Streamlit 

---

## 🏗️ Alcance de Ingeniería

La aplicación se enfoca en:

- Comportamiento **elástico lineal**
- Estructuras planas (2D)
- Idealización tipo **shear-building**
- Modelado equivalente lineal para aisladores sísmicos
- Uso educativo y comparativo (no destinado a diseño estructural final)

> ⚠️ **ISO-LATE no está diseñado para reemplazar software profesional de análisis no lineal avanzado**.

---

## 📐 Fundamento Teórico

La formulación principal se basa en:

- Análisis matricial de estructuras
- Ecuación de movimiento para sistemas de múltiples grados de libertad (MDOF):

$$
\mathbf{M}\ddot{\mathbf{u}} + \mathbf{C}\dot{\mathbf{u}} + \mathbf{K}\mathbf{u} = -\mathbf{M}\mathbf{r}\ddot{u}_g
$$

- **M** = matriz de masa del sistema  
- **C** = matriz de amortiguamiento  
- **K** = matriz de rigidez  
- **u** = vector de desplazamientos relativos respecto al terreno  
- **u̇** = vector de velocidades relativas  
- **ü** = vector de aceleraciones relativas  
- **u<sub>g</sub>̈** = aceleración del terreno (registro sísmico)  
- **r** = vector de influencia sísmica (usualmente un vector de unos que indica cómo la aceleración del terreno afecta a cada grado de libertad)

El término del lado derecho representa las fuerzas inerciales inducidas por la aceleración del terreno sobre la masa estructural.

---

### Interpretación Física

- El primer término representa las fuerzas inerciales internas.
- El segundo término corresponde a la disipación de energía por amortiguamiento.
- El tercer término describe la respuesta elástica del sistema.
- El término del lado derecho modela la excitación sísmica impuesta por el movimiento del suelo.

ISO-LATE resuelve esta ecuación utilizando:

- Superposición modal (para análisis espectral)
- Amortiguamiento de Rayleigh
- Integración numérica mediante el método de **Newmark-β**

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
```
---

## 📘 Manual de Usuario

ISO-LATE está estructurado en bloques secuenciales que permiten al usuario definir, analizar y comparar el comportamiento estructural de un sistema de base fija y uno con aislamiento sísmico.

Cada sección guía al usuario paso a paso a través del proceso completo de modelado y análisis dinámico.

---

### 🔹 1. Definición inicial del modelo

Al iniciar la aplicación, el usuario debe seleccionar el idioma de trabajo:  
- Inglés (**en**)  
- Español (**es**)  

Posteriormente, se definen los parámetros generales del modelo estructural:

- Geometría del sistema (número de niveles y vanos)
- Dimensiones de las secciones estructurales
- Propiedades mecánicas de los materiales
- Cargas consideradas en el análisis

Por defecto, las secciones se definen mediante dimensiones geométricas (por ejemplo, ancho y altura para secciones rectangulares).

Si se activa la opción **"Modo avanzado"**, las propiedades de los elementos se ingresan directamente mediante:

- Área transversal (A)
- Momento de inercia (I)

Este modo es especialmente útil cuando se modelan secciones no rectangulares o cuando se dispone directamente de propiedades obtenidas de catálogos estructurales.

<p align="center">
  <img src="assets/IMA1.png" width="900"/><br>
  <em>Figura 1 – Ventana inicial y definición de parámetros estructurales.</em>
</p>

---

### 🔹 2. Generación del modelo estructural

Una vez ingresados los parámetros, se presiona el botón **"Generar modelo estructural"**.

El programa construye automáticamente:

- Nodos
- Elementos
- Grados de libertad (GDL)

El modelo implementa un **diafragma rígido por piso**, lo que implica:

- Un único grado de libertad horizontal (UX) por nivel.
- Los grados de libertad verticales (UY) y rotacionales (θ) permanecen en cada nodo.

Posteriormente se realiza la **condensación de la matriz global**, reduciendo el sistema a un único grado de libertad horizontal por planta.

En esta sección el usuario puede visualizar:

- Matriz global de rigidez
- Matriz global de masa
- Matriz de transformación
- Matrices condensadas K y M

También se incluye una pestaña de **chequeo rápido de rigideces**, donde se comparan los valores obtenidos con la expresión aproximada:

$$
\frac{12EI}{L^3}
$$

<p align="center">
  <img src="assets/IMA2.png" width="900"/><br>
  <em>Figura 2 – Resumen del modelo y matrices condensadas.</em>
</p>

---

### 🔹 3. Espectro NEC-24 y Registro Sísmico

En esta sección se define el espectro de diseño conforme a la **Normativa Ecuatoriana de la Construcción NEC-24**.

El programa:

- Genera el espectro automáticamente.
- Muestra los coeficientes normativos utilizados.
- Indica aceleraciones espectrales relevantes (meseta y 1 segundo).

Además, se carga el registro sísmico para el análisis dinámico.

El usuario puede:

- Cargar archivos en formato **.TXT** o **.AT2**
- Utilizar un registro de ejemplo incorporado

Se admiten registros compatibles con:

- RENAC (Red Nacional de Acelerógrafos)
- PEER Ground Motion Database

El usuario puede optar por:

- Utilizar el registro crudo
- Aplicar filtrado y corrección de línea base

<p align="center">
  <img src="assets/IMA3.png" width="900"/><br>
  <em>Figura 3 – Espectro NEC-24 y carga del registro sísmico.</em>
</p>

---

### 🔹 4. Escalamiento del registro

El programa genera el espectro del registro ingresado y permite:

- Escalarlo al espectro NEC-24
- Seleccionar espectro elástico o inelástico
- Definir el amortiguamiento

<p align="center">
  <img src="assets/IMA4.png" width="900"/><br>
  <em>Figura 4 – Escalamiento del espectro.</em>
</p>

---

### 🔹 5. Diseño del Aislador LRB

El usuario puede diseñar un aislador tipo **LRB (Lead Rubber Bearing)** mediante:

**Método automático**
- Basado en ASCE 7 – Capítulo 17.

**Método por período objetivo**
- Define un período deseado (hasta 5 s).
- Advierte si el período no genera aislamiento efectivo.

Se obtienen:

- Rigidez efectiva
- Desplazamiento de diseño
- Energía disipada
- Propiedades lineales y no lineales
- Curva bilineal de histéresis

<p align="center">
  <img src="assets/IMA5.png" width="900"/><br>
  <em>Figura 5 – Diseño del aislador y curva de histéresis.</em>
</p>

---

### 🔹 6. Análisis Modal

Se ejecuta el análisis modal para:

- Sistema fijo
- Sistema aislado

Se muestran:

- Frecuencias naturales
- Períodos modales
- Matrices condensadas

<p align="center">
  <img src="assets/IMA6.png" width="900"/><br>
  <em>Figura 6 – Resultados modales.</em>
</p>

---

### 🔹 7. Formas Modales Normalizadas

Se representan gráficamente los modos de vibración normalizados junto con su período.

<p align="center">
  <img src="assets/IMA7.png" width="900"/><br>
  <em>Figura 7 – Modos de vibración.</em>
</p>

---

### 🔹 8. Esquema tipo Péndulo Invertido

Se presenta un resumen gráfico de:

- Rigideces por piso (Tf/m)
- Masas por piso (Tf·s²/m)

<p align="center">
  <img src="assets/IMA8.png" width="900"/><br>
  <em>Figura 8 – Esquema de masas y rigideces.</em>
</p>

---

### 🔹 9. Análisis Dinámico – Newmark-β

Se realiza el análisis en el dominio del tiempo para ambos sistemas.

El programa:

- Calcula amortiguamiento Rayleigh (5%)
- Obtiene coeficientes α y β
- Resuelve la ecuación de movimiento
- Genera respuestas en aceleración, velocidad y desplazamiento

Los resultados pueden exportarse en formato Excel.

<p align="center">
  <img src="assets/IMA9.png" width="900"/><br>
  <em>Figura 9 – Resultados dinámicos.</em>
</p>

---

### 🔹 10. Cortantes por Piso (RSA y THA)

Se pueden obtener cortantes mediante:

**Modal espectral (RSA)**
- Combinación modal SRSS.
- Basado en espectro NEC-24.

**Tiempo-historia (THA)**
- Valores instantáneos
- Máximo absoluto
- Máximos y mínimos

<p align="center">
  <img src="assets/IMA12.png" width="900"/><br>
  <em>Figura 12 – Cortantes por piso.</em>
</p>

---

### 🔹 11. Desplazamientos Laterales

Se obtienen desplazamientos mediante:

- Modal espectral
- Tiempo-historia

<p align="center">
  <img src="assets/IMA14.png" width="900"/><br>
  <em>Figura 14 – Desplazamientos laterales.</em>
</p>

---

### 🔹 12. Derivas de Entrepiso

Las derivas se calculan a partir de los desplazamientos obtenidos.

Se pueden visualizar dos tipos:

**Deriva real:**

$$
\displaystyle \frac{|\Delta|}{h}
$$

**Deriva NEC-24:**

$$
\displaystyle \frac{C_d\|\Delta|}{hI}
$$

<p align="center">
  <img src="assets/IMA16.png" width="900"/><br>
  <em>Figura 16 – Derivas por entrepiso.</em>
</p>

---

### 🔹 13. Comparativo Final – Base Fija vs Base Aislada

Se superponen:

- Cortantes
- Desplazamientos
- Derivas

Se presenta además un resumen cuantitativo de:

- Reducciones
- Incrementos
- Indicadores de eficiencia del aislamiento

<p align="center">
  <img src="assets/IMA17.png" width="900"/><br>
  <em>Figura 17 – Comparación final.</em>
</p>

---

### 🔹 Funcionalidades adicionales

- Cada parámetro incluye ayudas explicativas (ícono "?").
- Las gráficas pueden ampliarse.
- Los valores pueden copiarse directamente.
- Las tablas pueden exportarse en formato CSV o Excel.
