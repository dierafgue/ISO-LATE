# 💻 ISO-LATE  
### *Interactive Structural Analysis Tool – Fixed Base vs Base Isolation*

<p align="center">
  <img src="assets/logo.png" alt="ISO-LATE Logo" width="220"/>
</p>

<p align="center">
  <b>ISO-LATE</b> is an interactive engineering application developed to <b>simulate, analyze, and compare the seismic response of 2D structures</b> with <b>fixed-base</b> and <b>base-isolated systems</b>, supporting learning, research, and preliminary design in earthquake engineering.
</p>

<p align="center">
  🌐 <a href="https://iso-late.onrender.com" target="_blank"><b>Live App</b></a> • 📘 <a href="#user-manual">User Manual</a> • 🧠 <a href="#theoretical-background">Theory</a> • ⚙️ <a href="#installation">Installation</a>
</p>

---

## 📌 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Engineering Scope](#engineering-scope)
- [Theoretical Background](#theoretical-background)
- [Application Structure](#application-structure)
- [Installation](#installation)
- [Usage](#usage)
- [User Manual](#user-manual)
- [Validation & Limitations](#validation--limitations)
- [Technologies Used](#technologies-used)
- [Author & Academic Context](#author--academic-context)
- [License](#license)

---

## 🧭 Overview

**ISO-LATE** is designed as an educational and engineering-oriented tool that allows users to:

- Model **2D multi-storey frame structures**
- Perform **linear dynamic analysis**
- Compare **fixed-base** vs **base-isolated** structural behavior
- Visualize **seismic response metrics** in a clear and intuitive way

The application is especially oriented toward:
- Structural engineering students
- Earthquake engineering researchers
- Practicing engineers in early-stage design or concept validation

---

## ✨ Key Features

✔️ Parametric definition of 2D structures (stories, bays, geometry)  
✔️ Automatic generation of mass and stiffness matrices  
✔️ Modal analysis and response spectrum analysis  
✔️ Time-history analysis using **Newmark-β method**  
✔️ Base isolation modeling (LRB / NRB – linear equivalent)  
✔️ Side-by-side comparison: **Fixed vs Isolated**  
✔️ Clean and scalable engineering plots  
✔️ Web-based interactive interface (Streamlit)

---

## 🏗️ Engineering Scope

The application focuses on:

- **Linear elastic behavior**
- **Planar (2D) frame structures**
- **Shear-building idealization**
- **Equivalent linear modeling** for base isolators
- Educational and comparative purposes (not final design)

> ⚠️ *ISO-LATE is not intended to replace detailed nonlinear analysis software such as OpenSees, ETABS, or SAP2000.*

---

## 📐 Theoretical Background

The core formulation is based on:

- Matrix structural analysis
- Equation of motion for MDOF systems  

\[
\mathbf{M}\ddot{\mathbf{u}} + \mathbf{C}\dot{\mathbf{u}} + \mathbf{K}\mathbf{u} = -\mathbf{M}\mathbf{r} \ddot{u}_g
\]

- Modal superposition
- Rayleigh damping
- Newmark-β numerical integration
- Simplified equivalent linear base isolation theory

References include:
- Chopra, A.K. – *Dynamics of Structures*
- ASCE 7 / ASCE 41
- FEMA 440 / FEMA P-1050

---

## 🧩 Application Structure

```text
ISO-LATE/
│
├── app.py                     # Main Streamlit application
├── funciones_usuario.py       # Structural & dynamic analysis functions
├── requirements.txt           # Python dependencies
├── assets/                    # Images, logos, icons
│   └── logo.png
├── data/                      # Seismic records (optional)
└── README.md                  # Project documentation
