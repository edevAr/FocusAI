# FocusAI

Pipeline MLOps para clasificar entradas de diario como **Productivo** o **Procrastinación**, usando NLP, Ensemble Learning, MLflow, Airflow, FastAPI y Streamlit.

---

## Tabla de contenidos

1. [Qué hace el proyecto](#qué-hace-el-proyecto)
2. [Arquitectura](#arquitectura)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Requisitos previos](#requisitos-previos)
5. [Instalación paso a paso (para juniors)](#instalación-paso-a-paso-para-juniors)
6. [Correr el pipeline de entrenamiento](#correr-el-pipeline-de-entrenamiento)
7. [Levantar API + Frontend](#levantar-api--frontend)
8. [MLflow](#mlflow)
9. [Airflow](#airflow)
10. [Docker Compose](#docker-compose)
11. [API Reference](#api-reference)
12. [Credenciales de prueba](#credenciales-de-prueba)
13. [Troubleshooting](#troubleshooting)
14. [División del equipo: qué falta / qué mejorar](#división-del-equipo-qué-falta--qué-mejorar)

---

## Qué hace el proyecto

1. Lee un CSV de textos etiquetados (`Productivo` / `Procrastinación`).
2. Limpia el texto (tokenización, stopwords, lematización opcional con spaCy).
3. Vectoriza con **TF-IDF**.
4. Entrena y compara modelos ensemble (**XGBoost**, **Random Forest**, **LightGBM** / Gradient Boosting) con validación cruzada.
5. Registra métricas y el modelo en **MLflow**.
6. Expone predicciones con **FastAPI**.
7. Ofrece login + diario + gráfico histórico en **Streamlit**.
8. Orquesta el entrenamiento con un **DAG de Apache Airflow**.

---

## Arquitectura

```text
data/raw/journal_entries.csv
        │
        ▼
┌─────────────────── Apache Airflow DAG ───────────────────┐
│  extract_data → clean_data → feature_engineering         │
│       → train_model → evaluate_model                     │
│  (+ init_database en paralelo tras extract)              │
└──────────────────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
   Artefactos modelo              MLflow Tracking
   (vectorizer + classifier)      + Model Registry
        │
        ▼
   FastAPI (Docker)  ←→  SQLite (usuarios / diarios)
        │
        ▼
   Streamlit (login, caja de texto, Área de Procrastinación)
```

**DAG:** `focusai_productivity_pipeline`

| Task | Qué hace |
|------|----------|
| `extract_data` | Lee el CSV crudo |
| `clean_data` | NLP: limpia textos |
| `feature_engineering` | TF-IDF → matriz numérica |
| `train_model` | Compara modelos + log en MLflow |
| `evaluate_model` | Re-loguea Accuracy / F1 |
| `init_database` | Crea `data/database.db` |

---

## Estructura del repositorio

```text
FocusAI/
├── airflow/
│   └── dags/
│       └── productivity_pipeline_dag.py   # Orquestación
├── api/
│   ├── Dockerfile
│   ├── main.py                            # FastAPI
│   └── requirements-api.txt
├── config/
│   └── settings.py                        # Rutas y constantes
├── data/
│   ├── raw/
│   │   └── journal_entries.csv            # Dataset de ejemplo
│   ├── processed/                         # Generado al entrenar
│   ├── models/                            # vectorizer + modelo
│   └── database.db                        # SQLite (se genera local)
├── frontend/
│   └── app.py                             # Streamlit
├── scripts/
│   ├── run_pipeline.sh                    # Pipeline local sin Airflow
│   ├── start_mlflow.sh
│   └── start_airflow.sh
├── src/
│   ├── nlp/preprocess.py                  # Limpieza + TF-IDF
│   ├── training/train_model.py            # PyCaret/sklearn + MLflow
│   ├── training/predict.py                # Inferencia
│   └── database/db.py                     # Usuarios / Diarios
├── docker-compose.yml
├── requirements-dev.txt                   # Setup liviano (recomendado)
├── requirements.txt                       # Stack completo (Airflow+PyCaret)
└── README.md
```

---

## Requisitos previos

| Herramienta | Versión recomendada | Notas |
|-------------|---------------------|-------|
| Python | **3.10 – 3.12** | No uses 3.13/3.14 (rompe dependencias) |
| pip / venv | incluidos con Python | |
| Git | cualquier reciente | |
| Docker Desktop | opcional | Para `docker compose` |
| Homebrew (macOS) | opcional | `brew install libomp` si quieres LightGBM nativo |

> **Nota Windows:** LightGBM produce un `access violation` en Windows con datasets pequeños y se omite automáticamente. El pipeline corre con XGBoost + Random Forest + Gradient Boosting.

Comprueba tu Python:

```bash
python3.12 --version
# o
python3 --version
```

Si `python3` es 3.14, usa explícitamente `python3.12`.

---

## Instalación paso a paso (para juniors)

### 1. Clonar el repo

```bash
git clone https://github.com/edevAr/FocusAI.git
cd FocusAI
```

### 2. Crear entorno virtual

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

Deberías ver `(.venv)` al inicio de la terminal.

### 3. Instalar dependencias

**Opción A — recomendada para empezar (sin Airflow/PyCaret pesados):**

```bash
pip install -U pip
pip install -r requirements-dev.txt
```

**Opción B — stack completo del enunciado:**

```bash
pip install -r requirements.txt
# Opcional AutoML:
pip install "pycaret>=3.3.0,<3.4.0"
# Opcional lematización spaCy en español:
pip install spacy
python -m spacy download es_core_news_sm
```

### 4. Variables de entorno (opcional)

**macOS / Linux:**
```bash
export PYTHONPATH="$(pwd)"
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export FOCUSAI_API_URL="http://127.0.0.1:8000"
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:FOCUSAI_API_URL = "http://127.0.0.1:8000"
```

Tip: ejecuta el bloque de variables cada vez que abras una terminal nueva dentro del proyecto.

### 5. Verificar que importa

```bash
python -c "from src.nlp.preprocess import clean_text; print(clean_text('Hoy terminé mis tareas'))"
```

Si imprime texto limpio, estás listo.

---

## Correr el pipeline de entrenamiento

Esto ejecuta NLP → SQLite → entrenamiento → predicción de prueba:

```bash
chmod +x scripts/*.sh
./scripts/run_pipeline.sh
```

Equivalente manual:

**macOS / Linux:**
```bash
export PYTHONPATH="$(pwd)"
python -m src.nlp.preprocess
python -m src.database.db
python -m src.training.train_model
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m src.nlp.preprocess
python -m src.database.db
python -m src.training.train_model
```

### Artefactos que se generan

| Archivo | Descripción |
|---------|-------------|
| `data/processed/cleaned_entries.csv` | Textos limpios |
| `data/processed/vectorized_features.csv` | Matriz TF-IDF + etiqueta |
| `data/processed/metrics.json` | Métricas CV del mejor modelo |
| `data/processed/holdout_metrics.json` | Métricas sobre el hold-out (test set) |
| `data/processed/per_class_metrics.json` | Matriz de confusión + classification report |
| `data/models/tfidf_vectorizer.joblib` | Vectorizer para inferencia |
| `data/models/productivity_classifier.joblib` | Modelo entrenado y calibrado |
| `data/models/tuning_results.csv` | Tabla comparativa base vs. tuneado |
| `data/database.db` | Usuarios y diarios |
| `mlflow_tracking.db` | Base de datos MLflow local (SQLite) |

> Nota: el trainer intenta **PyCaret** primero. Si no está instalado, usa **sklearn** automáticamente y registra igual en MLflow.

---

## Levantar API + Frontend

Necesitas **dos terminales** (ambas con `.venv` activado y `PYTHONPATH`):

**Terminal 1 — API**

```bash
cd FocusAI
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Abre docs interactivas: http://127.0.0.1:8000/docs

**Terminal 2 — Streamlit**

```bash
cd FocusAI
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export FOCUSAI_API_URL=http://127.0.0.1:8000
streamlit run frontend/app.py
```

UI: http://127.0.0.1:8501

### Flujo en la web

1. Regístrate o inicia sesión.
2. Escribe una entrada de diario.
3. Pulsa **Clasificar**.
4. Mira el historial y el gráfico **Área de Procrastinación**.

---

## MLflow

```bash
./scripts/start_mlflow.sh
```

UI: http://127.0.0.1:5000

Si el server no está arriba, el entrenamiento cae a tracking local en `artifacts/mlruns/` (file store).

---

## Airflow

```bash
# Requiere dependencias de requirements.txt (apache-airflow)
./scripts/start_airflow.sh
```

- UI: http://127.0.0.1:8080  
- Usuario inicial: `admin` / `admin`  
- DAG a triggear manualmente: `focusai_productivity_pipeline`

El script configura `AIRFLOW_HOME` en `.airflow/` dentro del repo y apunta `DAGS_FOLDER` a `airflow/dags/`.

---

## Docker Compose

Con el modelo ya entrenado (o montando `data/`):

```bash
docker compose up --build
```

| Servicio | URL |
|----------|-----|
| MLflow | http://127.0.0.1:5000 |
| API | http://127.0.0.1:8000 |
| Streamlit | http://127.0.0.1:8501 |

---

## API Reference

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Healthcheck |
| POST | `/predict` | Clasifica texto; `usuario_id` opcional para guardar en DB |
| POST | `/predict/batch` | Clasifica lista de textos |
| POST | `/auth/register` | Alta de usuario |
| POST | `/auth/login` | Login |
| GET | `/users/{id}/diarios` | Historial de entradas |
| GET | `/users/{id}/procrastination-series` | Serie para el gráfico |

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"texto":"Hoy terminé el informe y avancé tres tareas clave."}'
```

Registro + login:

```bash
curl -X POST http://127.0.0.1:8000/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"ana","email":"ana@mail.com","password":"ana123"}'

curl -X POST http://127.0.0.1:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"ana","password":"ana123"}'
```

---

## Credenciales de prueba

Tras correr el pipeline / `python -m src.database.db`:

| Campo | Valor |
|-------|-------|
| Usuario | `demo` |
| Password | `demo123` |
| Email | `demo@focusai.local` |

---

## Troubleshooting

| Problema | Solución |
|----------|----------|
| `python` es 3.14 y pip falla | Usa `python3.12 -m venv .venv` |
| `No module named pkg_resources` | `pip install 'setuptools<81'` |
| LightGBM: `libomp.dylib` missing (macOS) | `brew install libomp` o ignóralo (cae a RF/XGBoost/GB) |
| LightGBM: `access violation` en Windows | Comportamiento conocido en Windows con datasets pequeños; se omite automáticamente |
| `Modelo no encontrado` al predecir | Corre el pipeline primero: `python -m src.nlp.preprocess` y `python -m src.training.train_model` |
| Streamlit no conecta a la API | Exporta `FOCUSAI_API_URL=http://127.0.0.1:8000` y verifica que uvicorn esté arriba |
| `ModuleNotFoundError: src...` | Linux/macOS: `export PYTHONPATH="$(pwd)"` — Windows PowerShell: `$env:PYTHONPATH = (Get-Location).Path` |
| `export` no reconocido en Windows | Usa PowerShell con `$env:VARIABLE = "valor"` en lugar de `export` |
| `ImportError: Blocked import of regex` (NLTK) | El venv está dentro del proyecto; el parche en `preprocess.py` lo resuelve automáticamente |
| PyCaret no instalado | Normal: hay fallback sklearn. Instálalo con `pip install "pycaret>=3.3.0,<3.4.0"` si el enunciado lo exige |
| Airflow no ve el DAG | Revisa que `AIRFLOW__CORE__DAGS_FOLDER` apunte a `airflow/dags` |

---

## División del equipo: qué falta / qué mejorar

El repo ya tiene un **esqueleto funcional end-to-end**. Abajo: estado actual y pendientes por rol (según el perfil del proyecto).

### 1. Edén — NLP Data Engineer

**Ya existe**
- Script de limpieza + TF-IDF (`src/nlp/preprocess.py`)
- Dataset de ejemplo en español (`data/raw/journal_entries.csv`)
- Persistencia de cleaned CSV, features y vectorizer

**Falta / mejorar**
- [ ] Ampliar el dataset real (idealmente cientos de ejemplos balanceados; hoy ~60)
- [ ] Integrar spaCy `es_core_news_sm` de forma obligatoria (hoy es opcional)
- [ ] Experimentar BoW vs TF-IDF y documentar comparación
- [ ] Pipeline de calidad de datos: duplicados, textos vacíos, balance de clases
- [ ] Tests unitarios de `clean_text` / `feature_engineering`
- [ ] Versionar datasets (DVC o carpeta `data/versions/`)

---

### 2. Luis Vasquez — Model Builder

**Estado:** ✅ **Completado al 100%**

**Entregables implementados**
- [x] **Hold-out train/test (80/20)**: Split estratificado fijo antes de cualquier entrenamiento. CV solo sobre el 80%; hold-out nunca visto durante training ni tuning.
- [x] **PyCaret como camino principal**: `compare_models(sort='F1', fold=CV_FOLDS)` + `tune_model()` con fallback automático a sklearn si PyCaret no está instalado.
- [x] **Tuneo de hiperparámetros**: `tune_model()` en PyCaret y `RandomizedSearchCV` en sklearn (20 iteraciones, `f1_weighted`). Tabla exportada a `data/models/tuning_results.csv` y parámetros logeados en MLflow con prefijo `tuned_`.
- [x] **Calibración de probabilidades**: `CalibratedClassifierCV(method='sigmoid', cv='prefit')` aplicado post-tuning. Platt scaling robusto para datasets pequeños (~48 muestras de entrenamiento).
- [x] **Métricas por clase**: Matriz de confusión + `classification_report` completo guardados en `data/processed/per_class_metrics.json`. F1/Precision/Recall por clase logeados en MLflow.
- [x] **Notebook EDA** (`notebooks/eda.ipynb`): Análisis completo — distribución de clases, longitud de texto, frecuencia de palabras, TF-IDF por clase, comparativa CV vs. hold-out, matriz de confusión y distribución de probabilidades calibradas.

**Métricas MLflow por run**

| Prefijo | Métricas |
|---------|---------|
| `cv_` | Accuracy, F1, Precision, Recall, AUC (validación cruzada) |
| `holdout_` | Accuracy, F1, Precision, Recall, AUC (test set fijo) |
| `f1_prod`, `f1_proc` | F1 por clase (Productivo / Procrastinación) |
| `tuned_*` | Hiperparámetros del modelo optimizado |

**Artefactos generados**

| Archivo | Descripción |
|---------|-------------|
| `data/processed/metrics.json` | Métricas CV del mejor modelo |
| `data/processed/holdout_metrics.json` | Métricas sobre el hold-out |
| `data/processed/per_class_metrics.json` | Matriz de confusión + classification report |
| `data/models/tuning_results.csv` | Tabla comparativa base vs. tuneado |
| `data/models/productivity_classifier.joblib` | Modelo final calibrado para serving |
| `notebooks/eda.ipynb` | Notebook EDA para la demo |

---

### 3. Luis Lanza — MLOps Tracker

**Ya existe**
- Logging de params/métricas/artefactos en MLflow
- Registro en Model Registry (`productivity_ensemble`)
- Script `scripts/start_mlflow.sh` + servicio en docker-compose

**Falta / mejorar**
- [ ] Documentar convención de experiments/runs/stages (`Staging` → `Production`)
- [ ] UI checklist: cómo promover un modelo en el Registry
- [ ] Alias de producción y carga desde Registry en la API (hoy carga joblib local)
- [ ] Comparación lado a lado de runs en la demo
- [ ] Backend store persistente (SQLite/Postgres) en vez de solo file store
- [ ] Alertas básicas si Accuracy/F1 bajan de un umbral

---

### 4. Flavio — Pipeline Orchestrator + DB

**Estado:** ✅ **Completado al 100%**  
*DAG de Airflow funcional, migraciones Alembic listas, base de datos SQLite persistente en WSL, y CRUD finalizado.*

**Entregables y Mejoras Implementadas**
- [x] **DAG de Airflow 100% Funcional**: Probado en WSL/Python 3.10 sin bloqueos, ejecutando todas las tareas (`extract_data`, `clean_data`, `wait_for_cleaned_csv`, `feature_engineering`, `train_model`, `evaluate_model`, `init_database`) con estado `SUCCESS`.
- [x] **PyCaret + compare_models en Modo Seguro WSL**: Re-integrado `compare_models(fold=5, sort='Accuracy')` operando con `n_jobs=1` y `log_experiment=False` para evitar congelamientos C++ / deadlocks en WSL.
- [x] **Migraciones Alembic**: Estructura de BD bajo versión `0001_create_usuarios_and_diarios.py` en sustitución de `CREATE TABLE IF NOT EXISTS`.
- [x] **Persistencia de Tracking MLflow**: Configurado para usar SQLite persistente en `/home/<usuario>/mlflow_tracking.db` evitando los "disk I/O errors" de `/mnt/c/`.
- [x] **Capa de Abstracción CRUD (`src/database/crud.py`)**: Funciones limpias de INSERT/SELECT para consumo de la UI/API sin requerir SQL crudo.
- [x] **Variables y Orquestación**: Archivo `.env.example` completo y script ejecutor `scripts/wsl_run_pipeline.sh`.
- [x] **Seed de Datos Sintéticos**: Inyección de 30 días de historial de demo simulando altibajos reales de productividad/procrastinación.

---

#### 🚀 Guía de Ejecución Rápida (Para el equipo)

Para levantar el entorno y ejecutar el pipeline completo en WSL:

1. **Abrir terminal WSL (Ubuntu)** y situarse en el proyecto:
   ```bash
   cd /mnt/c/Users/Gabo/Desktop/FocusAI
   ```
2. **Ejecutar el script bootstrap del pipeline**:
   ```bash
   bash scripts/wsl_run_pipeline.sh
   ```
   *Nota: El script inicializa la BD de Airflow en `/home/<usuario>/airflow-focusai`, configura las variables de 1 solo hilo (`OMP_NUM_THREADS=1`) para evitar bloqueos y ejecuta el DAG completo. MLflow guardará el tracking en `/home/<usuario>/mlflow_tracking.db`.*

---

#### 🔌 Instrucciones para Mireya (Frontend / Deployment)

La base de datos SQLite ya está inicializada con las migraciones de Alembic y dispone de una capa CRUD limpia. Para conectar FastAPI y Streamlit a la base de datos sin escribir SQL crudo, importa las funciones directamente desde `src/database/crud.py`:

```python
from src.database.crud import (
    registrar_usuario,
    obtener_usuario_por_email,
    guardar_diario,
    obtener_historial_diarios,
)

# 1. Registro de Usuario (retorna el ID creado)
user_id = registrar_usuario(
    nombre="nombre_usuario", 
    email="usuario@focusai.com", 
    password_hash=hash_de_contrasena
)

# 2. Login / Consulta por Email (retorna dict o None)
usuario = obtener_usuario_por_email("usuario@focusai.com")
# Output: {'id': 1, 'username': '...', 'email': '...', 'password_hash': '...', 'created_at': '...'}

# 3. Guardar Entrada de Diario (retorna el ID del diario)
diario_id = guardar_diario(
    usuario_id=user_id, 
    texto="Hoy estuve avanzando en la interfaz gráfica", 
    etiqueta_predicha="Productivo", 
    probabilidad=0.95
)

# 4. Obtener Historial de Diarios (retorna list[dict] ordenados por fecha)
historial = obtener_historial_diarios(usuario_id=user_id)
```


---

### 5. Mireya — Deployment & Serving

**Ya existe**
- FastAPI con predict/auth/historial
- Dockerfile de la API
- Streamlit con login/registro, caja de texto y área de procrastinación
- docker-compose (API + MLflow + frontend)

**Falta / mejorar**
- [ ] Usar `streamlit-authenticator` como pide el enunciado (hoy auth es custom vía API/SQLite)
- [ ] JWT o sesiones firmadas (passwords hoy con hash simple SHA256+salt)
- [ ] Healthchecks + restart policies más completos en Compose
- [ ] Cargar modelo desde MLflow Registry en runtime (no solo archivos locales)
- [ ] Mejorar UX del dashboard (filtros por fecha, export CSV)
- [ ] Tests de contrato API (`/predict`, `/auth/*`)
- [ ] Guía corta de demo en vivo (script de 3 minutos)

---

## Licencia / curso

Proyecto académico de pipeline MLOps (Ensemble + NLP). Ajustad la licencia si el equipo lo publica como open source.
