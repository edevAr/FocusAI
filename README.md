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

### Comparison and manual promotion

The supported demo path uses the Docker MLflow server and its persistent
`mlflow-data` volume. Train a candidate, compare its runs in the native MLflow
UI, and assign the lowercase aliases through MLflow itself:

```bash
docker compose --profile training run --rm pipeline
# Open http://127.0.0.1:5000, compare the candidate runs, then assign:
# staging -> the reviewed version; production -> the approved version.
```

FocusAI reads the `production` alias and quality-gate tags only. It does not
offer a promotion button, automatically change aliases, or prevent a privileged
MLflow operator from assigning an ineligible candidate. When that happens, the
read-only status panel keeps the observed alias, warning, and failed checklist
visible.

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

## Docker Compose (supported Linux path)

Requires Docker Engine with the Compose plugin on Linux. Copy the example
environment only when you need different thresholds or aliases:

```bash
cp .env.example .env
docker compose up --build
```

Compose starts MLflow, runs the Alembic migration, bootstraps demo data, then
starts the API and Streamlit. A fresh stack can be live while `/health/ready`
is degraded until a native MLflow `production` alias points to a loadable model.
Check the read-only state with:

```bash
curl http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/mlops/status
```

Restart without deleting managed data:

```bash
docker compose down
docker compose up -d
```

Do not use `down -v` when preserving the demo history. To roll back a model,
use the MLflow UI/API to reassign `production` to the previous version; to roll
back this status slice, revert only its API/frontend/DAG/documentation changes.
SQLite volumes are appropriate for the demo. A production migration to Postgres
requires a separate deployment plan and is not provisioned by this project.

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
| GET | `/health/live` | Proceso activo |
| GET | `/health/ready` | Schema, MLflow y bundle Production utilizables |
| GET | `/mlops/status` | Estado read-only: readiness, identidad Production, warnings y checklist |
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

**Estado:** ✅ Completado para la demo Docker/Linux

**Entregables implementados**
- [x] Logging de parámetros, métricas Accuracy/F1, warnings, artefactos e ID de run en MLflow.
- [x] Model Registry persistente para `productivity_ensemble`.
- [x] Convención documentada de aliases `staging` → `production`.
- [x] Checklist read-only y estado de calidad/modelo en FocusAI.
- [x] Carga de `models:/productivity_ensemble@production` desde la API.
- [x] Comparación de runs y promoción manual en la UI nativa de MLflow.
- [x] Backend SQLite persistente mediante volumen Docker.
- [x] Warnings de quality gate para Accuracy/F1 y elegibilidad visible.

**Decisiones de alcance**
- La promoción sigue siendo manual y la autoridad es MLflow.
- No hay alertas externas ni despliegue Postgres; son evoluciones futuras.

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

**Base heredada**
- FastAPI con predict/auth/historial
- Dockerfile de la API
- Streamlit con login/registro, caja de texto y área de procrastinación
- docker-compose (API + MLflow + frontend)

**Aporte de Mireya (implementado)**

*Autenticación con `streamlit-authenticator`* (lo que pedía el enunciado). El login
ahora lo gestiona la librería: renderiza el formulario, valida la contraseña con
**bcrypt** y mantiene la sesión con una **cookie firmada** (persiste al recargar).
El registro usa un formulario propio que persiste al usuario con hash **bcrypt** en
SQLite a través de la API. No se modificó el código de la BD de Flavio ni el flujo
`/auth/*` heredado (que sigue disponible para uso programático).

*UX del dashboard.* Filtro por **rango de fechas** en el Área de Procrastinación y
botones para **exportar CSV** tanto de la serie histórica como de las entradas.

*Tests de contrato de la API* en `tests/test_api_contract.py` (cubren `/predict`,
`/auth/login` y los `/auth/st/*`, con el modelo y la BD simulados vía monkeypatch).

**Endpoints nuevos para streamlit-authenticator**
- `POST /auth/st/register` — registra usuario con hash bcrypt (compatible con la librería).
- `GET /auth/st/credentials` — credenciales para `stauth.Authenticate` (solo hashes bcrypt).
- `GET /auth/st/user/{username}` — resuelve `id/username/email` tras autenticar (búsqueda insensible a mayúsculas, porque la librería normaliza el username a minúsculas).

**Archivos modificados/creados**
- `frontend/app.py` — login con streamlit-authenticator, sesión por cookie, filtros de fecha y export CSV.
- `api/main.py` — endpoints `/auth/st/*` (registro con bcrypt, credenciales y lookup).
- `src/database/crud.py` — helpers `listar_usuarios()` y `obtener_usuario_por_username()` (case-insensitive).
- `tests/test_api_contract.py` — tests de contrato de la API.
- `frontend/Dockerfile` — instala `streamlit-authenticator==0.3.2` y `PyYAML`.
- `api/requirements-api.txt` y `requirements-dev.txt` — añaden `bcrypt`.

**Cómo levantar y probar (Docker)**

```bash
# 1. Levantar el stack (MLflow + BD + API + frontend)
docker compose up --build

# 2. Entrenar y registrar el modelo (evita el paso de seed que rompe en este contenedor)
docker compose --profile training run --rm -e PYTHONPATH=/app pipeline \
  bash -lc "python -m src.nlp.preprocess && python -m src.training.train_model"

# 3. Asignar el alias 'production' (requerido para servir /predict)
docker compose exec api python -c "from mlflow.tracking import MlflowClient as M; c=M('http://mlflow:5000'); vs=[x for x in c.search_model_versions() if x.name=='productivity_ensemble']; v=max(vs, key=lambda x:int(x.version)).version; c.set_registered_model_alias('productivity_ensemble','production', v); print('production ->', v)"

# 4. Correr los tests
docker compose --profile training run --rm -e PYTHONPATH=/app pipeline \
  bash -lc "pip install -q bcrypt && pytest -q"
```

Web: http://127.0.0.1:8501 · API: http://127.0.0.1:8000/docs · MLflow: http://127.0.0.1:5000

**Notas / troubleshooting**
- **Puerto 5000 ocupado** en macOS: desactivar *AirPlay Receiver* (Ajustes → General → AirDrop y Handoff) o remapear el puerto de MLflow en `docker-compose.yml`.
- El alias `production` es un paso **manual** por diseño (el entrenamiento registra la versión pero no promueve aliases). También se puede asignar desde la UI de MLflow.
- El endpoint `/auth/st/credentials` solo expone usuarios con hash **bcrypt** (prefijo `$2`); los usuarios heredados con hash SHA-256 (p. ej. `demo`) no entran por streamlit-authenticator.

**Pendiente / mejoras futuras**
- [ ] JWT o sesiones firmadas para los endpoints `/auth/*` heredados (hoy SHA256+salt).
- [ ] Healthchecks + restart policies más completos en Compose.
- [ ] Cargar modelo desde MLflow Registry en runtime (no solo archivos locales).
- [ ] Guía corta de demo en vivo (script de 3 minutos).

---

## Licencia / curso

Proyecto académico de pipeline MLOps (Ensemble + NLP). Ajustad la licencia si el equipo lo publica como open source.
