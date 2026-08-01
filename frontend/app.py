"""
FocusAI — Frontend Streamlit
Login/Registro, caja de texto para diario y gráfico de procrastinación.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

API_URL = os.getenv("FOCUSAI_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="FocusAI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Fraunces:opsz,wght@9..144,600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Fraunces', serif !important; }
    .main {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 45%, #0ea5e9 160%);
    }
    .stApp { background: transparent; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_post(path: str, payload: dict) -> tuple[bool, dict | str]:
    try:
        resp = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
        data = resp.json() if resp.content else {}
        if resp.ok:
            return True, data
        return False, data.get("detail", resp.text)
    except requests.RequestException as exc:
        return False, str(exc)


def api_get(path: str) -> tuple[bool, dict | str]:
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=30)
        data = resp.json() if resp.content else {}
        if resp.ok:
            return True, data
        return False, data.get("detail", resp.text)
    except requests.RequestException as exc:
        return False, str(exc)


def init_session() -> None:
    if "user" not in st.session_state:
        st.session_state.user = None


def render_auth() -> None:
    st.title("FocusAI")
    st.caption("Clasifica tus entradas de diario: Productivo vs Procrastinación")

    tab_login, tab_register = st.tabs(["Iniciar sesión", "Registrarse"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submitted = st.form_submit_button("Entrar", use_container_width=True)
            if submitted:
                ok, data = api_post("/auth/login", {"username": username, "password": password})
                if ok:
                    st.session_state.user = data["user"]
                    st.success(f"Bienvenido, {data['user']['username']}")
                    st.rerun()
                else:
                    st.error(data)

    with tab_register:
        with st.form("register_form"):
            username = st.text_input("Usuario nuevo")
            email = st.text_input("Email")
            password = st.text_input("Contraseña nueva", type="password")
            submitted = st.form_submit_button("Crear cuenta", use_container_width=True)
            if submitted:
                ok, data = api_post(
                    "/auth/register",
                    {"username": username, "email": email, "password": password},
                )
                if ok:
                    st.success("Cuenta creada. Ahora inicia sesión.")
                else:
                    st.error(data)


def render_dashboard() -> None:
    user = st.session_state.user
    st.sidebar.title("FocusAI")
    st.sidebar.write(f"Hola, **{user['username']}**")
    if st.sidebar.button("Cerrar sesión"):
        st.session_state.user = None
        st.rerun()

    st.title("Diario de productividad")
    st.write("Escribe cómo fue tu día y el modelo Ensemble te clasificará.")

    col_input, col_result = st.columns([1.2, 1])

    with col_input:
        texto = st.text_area(
            "Entrada de diario",
            height=180,
            placeholder="Ej: Hoy terminé el informe, organicé mis tareas y avancé el pipeline MLOps...",
        )
        if st.button("Clasificar", type="primary", use_container_width=True):
            if not texto.strip():
                st.warning("Escribe algo primero.")
            else:
                ok, data = api_post(
                    "/predict",
                    {"texto": texto, "usuario_id": user["id"]},
                )
                if ok:
                    st.session_state.last_prediction = data
                else:
                    st.error(data)

    with col_result:
        pred = st.session_state.get("last_prediction")
        if pred:
            label = pred["prediccion"]
            color = "#22c55e" if label == "Productivo" else "#f97316"
            st.markdown(
                f"""
                <div style="padding:1.5rem;border-radius:16px;background:rgba(255,255,255,0.07);
                border:1px solid rgba(255,255,255,0.15);">
                    <p style="margin:0;opacity:.7;">Predicción</p>
                    <h2 style="margin:.2rem 0;color:{color};">{label}</h2>
                    <p style="margin:0;">Confianza:
                    {f"{pred['probabilidad']*100:.1f}%" if pred.get("probabilidad") is not None else "N/A"}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Texto limpio (NLP)"):
                st.code(pred.get("texto_limpio", ""), language=None)

    st.divider()
    st.subheader("Área de Procrastinación")
    st.caption("Histórico de entradas clasificadas por día")

    ok, data = api_get(f"/users/{user['id']}/procrastination-series")
    ok_d, diarios = api_get(f"/users/{user['id']}/diarios")

    if ok and isinstance(data, dict) and data.get("series"):
        df = pd.DataFrame(data["series"])
        fig = px.area(
            df,
            x="dia",
            y=["procrastinacion", "productivo"],
            title="Productivo vs Procrastinación",
            labels={"value": "Entradas", "dia": "Día", "variable": "Clase"},
            color_discrete_map={
                "procrastinacion": "#f97316",
                "productivo": "#22c55e",
            },
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Días registrados", len(df))
        c2.metric("Total procrastinación", int(df["procrastinacion"].sum()))
        c3.metric("Total productivos", int(df["productivo"].sum()))
    else:
        st.info("Aún no hay historial. Clasifica tu primera entrada.")

    if ok_d and isinstance(diarios, dict) and diarios.get("diarios"):
        st.subheader("Últimas entradas")
        hist = pd.DataFrame(diarios["diarios"])
        show = hist[["created_at", "prediccion", "probabilidad", "texto"]].sort_values(
            "created_at", ascending=False
        )
        st.dataframe(show, use_container_width=True, hide_index=True)


def main() -> None:
    init_session()
    # Health check discreto
    try:
        requests.get(f"{API_URL}/health", timeout=3)
    except requests.RequestException:
        st.warning(
            f"No se pudo conectar a la API en `{API_URL}`. "
            "Levanta FastAPI con `uvicorn api.main:app --reload`."
        )

    if st.session_state.user is None:
        render_auth()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
