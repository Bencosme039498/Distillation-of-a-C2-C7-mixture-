# ⚗️ Ethane–Heptane Phase Equilibrium — Python Numerical Modeling

Python-based numerical modeling project developed to analyze the vapor–liquid equilibrium behavior of an ethane–n-heptane binary mixture and generate its phase equilibrium diagram.


## 🔎 Project Overview

This project applies Python and numerical methods to model the thermodynamic behavior of an ethane–n-heptane binary mixture.

The program performs iterative vapor–liquid equilibrium calculations across a range of temperatures and uses the resulting liquid and vapor compositions to construct a phase diagram.

The analysis combines thermodynamic calculations, numerical linear algebra, iterative computation, and scientific visualization within a single Python workflow.

## 🎯 Project Objectives

The project was designed to:

- Model the vapor–liquid equilibrium of an ethane–n-heptane mixture
- Calculate thermodynamic properties across different temperatures
- Determine liquid and vapor phase compositions
- Solve systems of equations numerically
- Calculate polynomial roots for compressibility factors
- Perform iterative equilibrium calculations
- Calculate mixture enthalpy and entropy properties
- Generate a phase equilibrium diagram using Python

## 🛠️ Tools & Libraries

- **Python** — Scientific and numerical programming
- **NumPy** — Arrays, numerical calculations, linear algebra, and polynomial roots
- **Matplotlib** — Scientific data visualization
- **NumPy Linear Algebra** — Numerical solution of systems of equations

## 🧮 Numerical Methodology

The Python model performs calculations over **50 temperature points from 298 K to 400 K**.

For each temperature, the program:

1. Calculates reduced temperature and pressure
2. Estimates saturation pressures and model parameters
3. Initializes liquid and vapor compositions
4. Solves systems of equations using `numpy.linalg.solve`
5. Calculates polynomial roots to determine compressibility factors
6. Evaluates liquid and vapor fugacity coefficients
7. Updates equilibrium constants and phase compositions iteratively
8. Calculates thermodynamic properties including enthalpy and entropy
9. Stores the calculated results for phase-diagram construction

## 📈 Phase Equilibrium Visualization

The calculated liquid and vapor compositions are evaluated across the temperature range and visualized using Matplotlib.

The resulting diagram illustrates the relationship between temperature and the calculated ethane composition in the liquid and vapor phases.

![Ethane-Heptane Phase Diagram](ethane-heptane-phase-diagram.png)

## 💡 Skills Demonstrated

This project demonstrates the application of Python to a complex quantitative engineering problem, including:

- Numerical analysis
- Array-based calculations
- Linear algebra
- Polynomial root solving
- Iterative algorithms
- Scientific computing
- Mathematical modeling
- Data visualization

## 📁 Project File

`ethane_heptane_phase_equilibrium.py` — Python script containing the numerical calculations, iterative equilibrium model, thermodynamic calculations, and phase-diagram generation.

## 👤 Author

**Juan Alejandro Bencosme Diaz**  
Data Analyst | Power BI | Python | Data Visualization
