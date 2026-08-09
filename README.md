# FocusAI — Documento de Diseño Técnico

> **Estado:** Revisado | **Fecha:** Agosto 2026 | **Proyecto:** Pipeline MLOps — Ensemble Learning + NLP

## Documento de Diseño Técnico

> **Estado del documento:** Revisado | **Fecha:** Agosto 2026

### Resumen

FocusAI es un sistema MLOps que permite a los usuarios registrar entradas de diario en texto libre y, mediante un modelo de Inteligencia Artificial basado en Ensemble Learning y Procesamiento de Lenguaje Natural (NLP), clasifica automáticamente si el día del usuario fue **"Productivo"** o de **"Procrastinación"**. El objetivo principal es brindar retroalimentación instantánea y visualización histórica sobre los patrones de comportamiento del usuario a través de un panel de control interactivo, automatizando todo el ciclo de vida del aprendizaje automático, desde la ingesta de datos hasta el despliegue del modelo en un entorno productivo.

### Supuestos

- Los textos ingresados por los usuarios estarán redactados principalmente en **idioma español**, lo que requiere un pipeline NLP ajustado a dicho idioma.
- El despliegue inicial (MVP) se diseñará para operar en entornos locales aislados mediante contenedores (Docker) y subsistemas de Linux (WSL), utilizando **SQLite** como motor de base de datos para simplificar la persistencia.
- El pipeline tolerará limitaciones del hardware subyacente, desactivando algoritmos conflictivos (como LightGBM en entornos Windows pequeños) de forma automática.

### Alcance y Fases

**Fase 1 (Alcance Actual del Proyecto):**
- Ingesta, limpieza y balanceo de datos (256 muestras sintéticas).
- Orquestación automatizada de tareas (extracción, procesamiento, entrenamiento y evaluación) utilizando **Apache Airflow**.
- Entrenamiento, comparación y calibración de modelos Ensemble (XGBoost, Random Forest, Gradient Boosting) usando **PyCaret**.
- Registro y versionado de modelos (Tracking) usando **MLflow**.
- Despliegue de una API REST mediante **FastAPI** para inferencia en tiempo real.
- Desarrollo de un Frontend interactivo (Login, ingreso de texto y gráficos históricos) usando **Streamlit**.

**Fuera de Alcance:**
- Migración de la base de datos a sistemas distribuidos como PostgreSQL o MySQL.
- Despliegue de los artefactos en proveedores de nube pública (AWS SageMaker, Azure ML).
- Autenticación programática basada en tokens JWT para endpoints externos.

---

### 1. Requerimientos

#### 1.1 Requerimientos Funcionales

- **Clasificación NLP:** El sistema debe procesar texto libre (limpieza, tokenización, lematización con spaCy) y extraer características numéricas utilizando el algoritmo TF-IDF.
- **Experimentación de ML Automatizada:** El sistema debe comparar automáticamente el rendimiento de múltiples modelos y seleccionar el mejor (basado en métrica F1-Score) sin intervención manual durante el pipeline.
- **Inferencia en Tiempo Real:** Los usuarios deben poder enviar texto a una API y recibir la clasificación (Productivo/Procrastinación) y su probabilidad calibrada en formato JSON.
- **Persistencia y UI:** Los usuarios deben poder autenticarse de forma segura, registrar sus textos clasificados y visualizar gráficamente su historial de comportamiento a través de un panel web (Área de Procrastinación).

#### 1.2 Requerimientos No Funcionales

- **Reproducibilidad:** El entorno debe ser altamente reproducible, gestionando dependencias estrictas (Python 3.10–3.12) y empaquetado a través de Docker Compose.
- **Trazabilidad:** Cada ejecución de entrenamiento debe ser registrada meticulosamente en MLflow, almacenando hiperparámetros, métricas (Accuracy, F1, Recall, Precision, AUC) y los artefactos del modelo ganador.
- **Eficiencia y Seguridad en WSL:** Las operaciones en base de datos deben utilizar una capa CRUD abstracta con SQLAlchemy/Alembic en lugar de SQL crudo para prevenir inyecciones; las ejecuciones de modelado deben limitar la concurrencia a un solo hilo (`n_jobs=1`) para prevenir deadlocks en WSL.

#### 1.3 Estimación de Capacidad

| Parámetro | Estimación |
|---|---|
| Consultas por Segundo (QPS) | < 5 durante uso normal del frontend |
| Latencia Esperada | < 300 ms por solicitud de la API |
| Volumen de Almacenamiento | SQLite local sin degradación de rendimiento para el volumen inicial proyectado |

---

### 2. Entidades Principales

| Entidad | Campos |
|---|---|
| **Usuarios** | `id`, `username`, `email`, `password_hash` (bcrypt), `created_at` |
| **Entradas de Diario** | `id`, `usuario_id` (FK), `texto`, `etiqueta_predicha`, `probabilidad`, `created_at` |
| **Artefactos del Modelo** | `tfidf_vectorizer.joblib`, `productivity_classifier.joblib` — versionados por MLflow |

---

### 3. API del Sistema (FastAPI)

El backend expone una arquitectura RESTful con soporte OpenAPI.

| Método | Ruta | Descripción | Input | Output |
|---|---|---|---|---|
| `GET` | `/health/ready` | Valida la disponibilidad del modelo (`production`) en MLflow y el esquema de BD | N/A | `200 OK` / `503` |
| `POST` | `/predict` | Clasifica un texto individual; si recibe `usuario_id`, almacena la predicción | `{"texto": str}` | `{"prediccion": str, "probabilidad": float}` |
| `POST` | `/auth/register` | Crea un usuario con contraseña hasheada | `{"username": str, "email": str, "password": str}` | `201 Created` |
| `POST` | `/auth/login` | Autentica a un usuario programáticamente | `{"username": str, "password": str}` | `200 OK` (Token/Auth) |
| `GET` | `/users/{id}/procrastination-series` | Obtiene el historial formateado para graficar | Parámetro URL `id` | `List[Dict]` de fechas y resultados |

---

### 4. Flujo de Datos Arquitectónico

La arquitectura bifurca el sistema en dos canales independientes:

#### Canal de Entrenamiento (Orquestado por Airflow)

```
journal_entries.csv
        │
        ▼
  [extract_data]  ──────────────────────────────┐
        │                                        │
        ▼                                  [init_database]
   [clean_data]                                  │
   (NLP: stopwords, lematización spaCy)          ▼
        │                               SQLite + Alembic
        ▼
[feature_engineering]
   (TF-IDF Vectorización)
        │
        ▼
  [train_model]
   (PyCaret compare_models, 5-Fold CV)
        │
        ▼
 [evaluate_model]
   (Métricas → MLflow SQLite)
```

#### Canal de Inferencia y UX

```
Usuario (Streamlit)
        │  ingresa texto
        ▼
   FastAPI /predict
        │  carga Pipeline sklearn (Vectorizer + Clasificador)
        ▼
   Predicción JSON
        │
        ├─── INSERT en SQLite (diarios)
        │
        └─── Streamlit renderiza respuesta
             + actualiza Área de Procrastinación
```

---

### 5. Diseño del DAG de Airflow

El orquestador automatiza la tubería de ML mediante el DAG `focusai_productivity_pipeline`.

**Topología de las tareas:**

| Tarea | Descripción |
|---|---|
| `extract_data` | Ingiere y valida el formato del CSV |
| `init_database` *(paralela)* | Asegura que la BD SQLite y tablas existan vía migraciones Alembic |
| `clean_data` | Aplica NLP avanzado: normalización, tokenización, lematización y exporta `data_quality_report.json` |
| `feature_engineering` | Convierte texto en matriz dispersa mediante TF-IDF |
| `train_model` | Emplea `compare_models` de PyCaret con 5 K-Folds; loguea hiperparámetros en MLflow (`sqlite:////home/<user>/mlflow_tracking.db`) |
| `evaluate_model` | Genera reporte final con métricas del dataset de hold-out y finaliza el pipeline |

**Configuración de resiliencia del DAG:**

```python
default_args = {
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "email_on_failure": True,
}
```

---

### 6. Inmersiones Profundas

#### 6.1 Procesamiento NLP y Calidad de Datos

Para garantizar un aprendizaje libre de sesgo, la tubería incluye reportes de calidad que detectan y purgan duplicados. La lematización se impuso con la librería **spaCy** frente a NLTK, ya que la extracción de características con TF-IDF alcanzó resultados de clasificación (**F1-Score: ~0.91**) notablemente superiores a un modelo base, permitiendo al sistema identificar ponderaciones clave en el vocabulario productivo/procrastinador.

#### 6.2 Tracking, Versionado y Alias de MLflow

- **Datasets:** A través de un script, se crean snapshots cifrados en SHA256 dentro de `data/versions/` (actualmente en `v1.1.0`), previniendo corrupción de los datos origen.
- **MLflow Model Registry:** Los modelos exitosos se envían al repositorio local. El despliegue a la API requiere que el equipo, previa evaluación en la UI de MLflow (puerto 5000), asigne manualmente los alias `staging` (revisión) y `production` (aprobado y servido por FastAPI).
- **Backend de tracking:** `sqlite:////home/<usuario>/mlflow_tracking.db` — en filesystem nativo Linux para evitar errores de I/O en `/mnt/c/`.

#### 6.3 Seguridad y Persistencia (CRUD)

Se abandonó el uso de SQL crudo y se implementó una abstracción CRUD en `src/database/crud.py`. Todas las contraseñas que transitan a través de los endpoints `/auth/*` son cifradas utilizando **bcrypt**, vital ya que la sesión persistente del usuario se gestiona mediante cookies firmadas gestionadas por `streamlit-authenticator`.

```python
# Importación para Mireya (Frontend / API)
from src.database.crud import (
    registrar_usuario,
    obtener_usuario_por_email,
    guardar_diario,
    obtener_historial_diarios,
)
```

#### 6.4 Metodología de Pruebas (Testing)

| Suite | Archivo | Descripción |
|---|---|---|
| **Unitarias (NLP)** | `tests/test_nlp.py` | Comprueba funciones de limpieza de texto, detección de calidad y eficacia matemática del vectorizador |
| **Integración (API Contract)** | `tests/test_api_contract.py` | Valida los códigos HTTP en las rutas de predicción y autenticación, empleando BD y modelos simulados (Monkeypatch) |

---

### 7. Decisiones de Diseño Técnico (TDDs)

#### TDD-1: Estrategia de Vectorización — BoW vs TF-IDF

> **Problema:** Determinar cómo convertir los textos libres del usuario en variables predictivas significativas para los árboles de decisión.

| | Opción 1: Bag of Words (BoW) | Opción 2: TF-IDF *(Elegida)* |
|---|---|---|
| **Pros** | Simple de implementar, cálculo rápido | Pondera a la baja palabras ubicuas; da relevancia estadística a palabras clave únicas |
| **Contras** | No distingue importancia contextual; palabras comunes opacan a las críticas | Genera matrices de características más dispersas |

**Conclusión:** Se eligió TF-IDF. La integración produjo un incremento directo en las métricas de Precision y Recall. La comparativa técnica se exportó a `docs/bow_vs_tfidf.md`.

#### TDD-2: Mitigación de Congelamientos en WSL / PyCaret

> **Problema:** La integración original del orquestador sufría bloqueos abruptos en WSL durante la optimización paralela de PyCaret.

| | Opción 1: Deshabilitar algoritmos pesados | Opción 2: Ejecución Serial en Modo Seguro *(Elegida)* |
|---|---|---|
| **Pros** | Reduce carga en memoria | Resuelve deadlocks a nivel C++ de forma definitiva sin sacrificar modelos |
| **Contras** | Reduce capacidad de comparación entre modelos | La tarea `train_model` toma ligeramente más tiempo |

**Configuración implementada:**

```python
# src/training/train_model.py — inyectadas ANTES de cualquier import numérico
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

# PyCaret setup en Modo Seguro WSL
setup(
    data=data, target="etiqueta",
    n_jobs=1,           # PROHIBIDO >1: deadlock C++ en WSL
    log_experiment=False,  # MLflow se loguea manualmente
    html=False,
    verbose=False,
)
best_model = compare_models(fold=5, sort="Accuracy", verbose=False)
```

**Conclusión:** La Opción 2 resolvió los deadlocks. La tubería Airflow es **100% resiliente** y completa su ejecución sin interrumpir la persistencia en `database.db`.

---

### 8. Guía Detallada de Ejecución y Despliegue

Esta sección detalla los pasos exactos para configurar el entorno, ejecutar la tubería de datos y levantar los servicios de inferencia y la interfaz de usuario, ya sea de forma local (WSL/macOS) o mediante contenedores.

#### 8.1 Requisitos Previos

Antes de iniciar, el sistema anfitrión debe cumplir con las siguientes dependencias:

| Herramienta | Requisito | Notas |
|---|---|---|
| **Python** | 3.10 – 3.12 obligatorio | No usar 3.13 o 3.14: rompen dependencias internas |
| **pip / venv** | Incluidos con Python | Gestión del entorno aislado |
| **Git** | Cualquier versión reciente | Clonado del repositorio |
| **Docker Desktop** | Opcional | Para despliegue en contenedores |
| **Homebrew (macOS)** | Opcional | `brew install libomp` requerido para LightGBM nativo |

#### 8.2 Preparación del Entorno Local

**Paso 2.1 — Clonar y crear entorno virtual**

Descarga el repositorio y aísla el entorno de desarrollo:

```bash
git clone https://github.com/edevAr/FocusAI.git
cd FocusAI
python3.12 -m venv .venv
source .venv/bin/activate
```

> Para Windows PowerShell, la activación se realiza con `.venv\Scripts\activate`.

**Paso 2.2 — Instalación de dependencias**

Instala los requerimientos básicos y descarga el modelo de lenguaje en español de spaCy, obligatorio para la lematización:

```bash
pip install -U pip
pip install -r requirements-dev.txt
python -m spacy download es_core_news_sm
```

> Para instalar el stack completo que incluye Airflow y PyCaret, usar `requirements.txt`.

**Paso 2.3 — Configuración de Variables de Entorno**

Exporta las rutas necesarias para que Python y Streamlit localicen los módulos y la API:

```bash
export PYTHONPATH="$(pwd)"
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export FOCUSAI_API_URL="http://127.0.0.1:8000"
```

#### 8.3 Ejecución del Pipeline de Entrenamiento

El entrenamiento procesa el texto (NLP), inicializa la base de datos (SQLite) y entrena el modelo. Para ejecutar el pipeline completo de un solo golpe:

```bash
chmod +x scripts/*.sh
./scripts/run_pipeline.sh
```

**Ejecución Manual Modular** (paso a paso):

```bash
python -m src.nlp.preprocess
python -m src.database.db
python -m src.training.train_model
```

**Artefactos generados:**

| Archivo | Descripción |
|---|---|
| `data/processed/cleaned_entries.csv` | Textos limpios post-NLP |
| `data/processed/vectorized_features.csv` | Matriz TF-IDF + etiqueta |
| `data/processed/metrics.json` | Métricas CV del mejor modelo |
| `data/processed/holdout_metrics.json` | Métricas sobre el hold-out (test set) |
| `data/processed/per_class_metrics.json` | Matriz de confusión + classification report |
| `data/models/tfidf_vectorizer.joblib` | Vectorizer TF-IDF para inferencia |
| `data/models/productivity_classifier.joblib` | Modelo entrenado y calibrado |
| `data/database.db` | Base de datos SQLite (usuarios + diarios) |
| `~/mlflow_tracking.db` | Backend de tracking MLflow (SQLite en Linux nativo) |

#### 8.4 Despliegue de Servicios (API y Frontend)

El sistema requiere levantar el backend y el frontend en **dos terminales distintas**, ambas con `.venv` activado y `PYTHONPATH` exportado.

**Terminal 1 — FastAPI (Backend)**

Levanta el servidor Uvicorn en el puerto 8000:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Documentación interactiva disponible en: `http://127.0.0.1:8000/docs`

**Terminal 2 — Streamlit (Frontend)**

Configura la URL de la API y arranca la interfaz gráfica:

```bash
export FOCUSAI_API_URL=http://127.0.0.1:8000
streamlit run frontend/app.py
```

Interfaz de usuario disponible en: `http://127.0.0.1:8501`

#### 8.5 Alternativa Automatizada: Docker Compose

Para entornos basados en Linux con Docker Engine, el sistema completo puede levantarse sin configuraciones manuales.

**Paso 5.1 — Levantar el Stack**

Copia el entorno de ejemplo y construye los contenedores:

```bash
cp .env.example .env
docker compose up --build
```

| Servicio | URL |
|---|---|
| MLflow | `http://127.0.0.1:5000` |
| API | `http://127.0.0.1:8000` |
| Streamlit | `http://127.0.0.1:8501` |

**Paso 5.2 — Entrenamiento y Asignación de Alias (Docker)**

Para que la API tenga un modelo disponible, ejecuta el entrenamiento dentro del contenedor y asigna el alias `production` al modelo ganador:

```bash
# 1. Ejecutar el pipeline de entrenamiento
docker compose --profile training run --rm -e PYTHONPATH=/app pipeline \
  bash -lc "python -m src.nlp.preprocess && python -m src.training.train_model"

# 2. Asignar el alias 'production' al modelo
docker compose exec api python -c "
from mlflow.tracking import MlflowClient as M
c = M('http://mlflow:5000')
vs = [x for x in c.search_model_versions() if x.name == 'productivity_ensemble']
v = max(vs, key=lambda x: int(x.version)).version
c.set_registered_model_alias('productivity_ensemble', 'production', v)
print('production ->', v)
"
```

> **Nota:** El alias `production` es un paso **manual** por diseño. El entrenamiento registra la versión pero no promueve aliases automáticamente. También se puede asignar desde la UI de MLflow en `http://127.0.0.1:5000`.

#### 8.6 Credenciales de Prueba del Sistema

Una vez inicializada la base de datos (vía script o Docker), el sistema genera un usuario predeterminado con datos históricos inyectados para probar el dashboard:

| Campo | Valor |
|---|---|
| **Usuario** | `demo` |
| **Password** | `demo123` |
| **Email** | `demo@focusai.local` |

---

## Licencia / curso

Proyecto académico de pipeline MLOps (Ensemble + NLP). Ajustad la licencia si el equipo lo publica como open source.
