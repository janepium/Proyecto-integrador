# Detección Binaria de Intrusiones en Redes mediante Aprendizaje Automático Clásico y Cuántico (QSVM) sobre CICIDS-2017

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-AER%20GPU-6929C4?logo=qiskit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20Opt-2E7D32)

Repositorio oficial del proyecto de investigación desarrollado en el programa de **Ingeniería de Sistemas** de la **Universidad de Antioquia** (Medellín, Colombia). Este trabajo evalúa el desempeño discriminativo y la viabilidad computacional de una **Máquina de Vectores de Soporte Cuántica híbrida (QSVM)** frente a referentes clásicos (**Random Forest** y **SVM-RBF**) para la detección binaria de intrusiones de red bajo un flujo estricto libre de filtración de datos (*data leakage*).

---

## Autores

- **Miguel Puerta** — `miguel.puerta@udea.edu.co`
- **Angelo Gallego** — `angelo.gallego@udea.edu.co`
- **Sarah Yauripoma** — `sarah.yauripomac@udea.edu.co`
- **Valentina Hoyos Castrillón** — `valentina.hoyos1@udea.edu.co`

---

##  Metodología y Arquitectura Experimental

Para garantizar métricas auditables y evitar el sobreajuste artificial, el flujo experimental sigue un orden estricto:

1. **Limpieza y Partición Estratificada:** De los **2,520,798 registros limpios** del conjunto CICIDS-2017 (tráfico benigno vs. ataque), se realiza primero la división estratificada **70% entrenamiento / 15% validación / 15% prueba** antes de aplicar cualquier transformación estadística.
2. **Preprocesamiento sin Filtración de Datos:** Ajustado exclusivamente sobre la partición de entrenamiento:
   - Filtrado de alta correlación ($\vert{}r\vert{} > 0.95$), conservando **46 variables** relevantes.
   - Transformación logarítmica para mitigar asimetrías en distribuciones de tráfico.
   - Escalamiento estándar y reducción dimensional mediante **PCA a 8 componentes principales** ($82.5\%$ de varianza explicada) para igualar la dimensionalidad de entrada al circuito cuántico.
3. **Submuestreo Estratificado y Codificación Cuántica:** Submuestreo estratificado ($N_{\text{entrenamiento}}=10,000$, $N_{\text{validación}}=1,000$, $N_{\text{prueba}}=1,000$), normalización Min-Max en $[0, \pi]$ y codificación mediante un circuito variacional `ZZFeatureMap` de **8 qubits** (profundidad 39, 111 compuertas) simulado con **Qiskit AER** (aceleración GPU) y optimizado mediante búsqueda bayesiana con **Optuna**.

---

##  Resumen de Resultados ($N_{\text{prueba}} = 1,000$)

### Métricas de Clasificación sobre la Clase Positiva (Ataque)

| Modelo | Espacio de Características | Precisión | *Recall* | *F1-Score* |
| :--- | :---: | :---: | :---: | :---: |
| **RF** | 46 variables | **0.9971** | **0.9942** | **0.9971** |
| **RF con PCA** | 8 componentes | 0.9760 | 0.9645 | 0.9702 |
| **SVM-RBF** | 8 componentes | 0.9765 | 0.9822 | 0.9794 |
| **QSVM** | 8 componentes | 0.9822 | 0.9822 | 0.9822 |

### Discriminación Global, Falsos Positivos y Tiempos de Cómputo

| Modelo | ROC-AUC | AUPRC | FPR (%) | T. Entrenamiento (s) | T. Inferencia (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RF** | **1.0000** | **1.0000** | **0.00%** | 0.642 | **0.090** |
| **RF con PCA** | 0.9992 | 0.9962 | 0.48% | 0.671 | 0.098 |
| **SVM-RBF** | 0.9925 | 0.9826 | 0.48% | **0.628** | 0.122 |
| **QSVM** | 0.9970 | 0.9840 | 0.36% | 961.371 | 160.482 |

> **Hallazgo clave:** **RF** sobre las 46 características es la opción más robusta y eficiente para despliegue operativo en tiempo real. Sin embargo, bajo igualdad de condiciones en el subespacio comprimido de 8 componentes PCA, la **QSVM** supera tanto al **SVM-RBF** como al **RF con PCA**, evidenciando una mayor capacidad discriminativa en espacios compactos pese al cuello de botella cuadrático $O(N^2)$ de la simulación cuántica.

---

## Estructura del Repositorio

```text
├── experimento_ids_clasico_cuantico.ipynb   # Notebook completo con preprocesamiento, modelos y evaluación
├── requirements.txt                         # Dependencias de Python necesarias para ejecutar el proyecto
└── README.md                                # Documentación general del proyecto
```

> **Nota sobre los datos:** Este repositorio no incluye los archivos `.csv` de **CICIDS-2017** (*MachineLearningCVE*) debido a su alto peso (varios GB). Al ejecutar el notebook, los archivos se leen directamente desde el almacenamiento del entorno (Kaggle) o desde una carpeta local configurada por el usuario.

---

## Cómo reproducir los resultados

### Opción 1: Kaggle / Google Colab
Abrir el notebook principal (`experimento_ids_clasico_cuantico.ipynb`) en **Kaggle Notebooks** o **Google Colab**, añadir el dataset **CICIDS-2017** en los datos de entrada, activar el acelerador **GPU** y ejecutar **Entorno de ejecución → Ejecutar todas** (*Run All*). Si el dataset no está disponible localmente, el notebook detecta la ruta de entrada de Kaggle (`/kaggle/input/cicids2017/`) e instala `qiskit` / `qiskit-aer-gpu` si hace falta.

### Opción 2: local
```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook experimento_ids_clasico_cuantico.ipynb
```
*(Nota: Para ejecución local, colocar previamente los 8 archivos `.csv` de CICIDS-2017 en una carpeta local y actualizar la ruta en la celda de carga).*

**Tiempo de ejecución:** $\approx 20\text{ minutos}$ con aceleración **GPU** (el cálculo de la matriz del kernel cuántico $O(N^2)$ y la optimización bayesiana con Optuna son las etapas más costosas). Todas las semillas están fijas (`SEMILLA = 42`), por lo que los resultados son reproducibles.
