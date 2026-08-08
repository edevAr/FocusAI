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


def status_messages(status: dict) -> dict[str, str | list[str]]:
    readiness = status.get("readiness", {})
    state = readiness.get("status", "unknown")
    causes = ", ".join(readiness.get("causes", {}).values())
    production = status.get("production")
    identity = (
        f"{production['name']}@{production['alias']} (version {production['version']})"
        if production
        else "Production alias unavailable"
    )
    checklist = status.get("checklist", {})
    return {
        "readiness": f"Readiness: {state}" + (f" — {causes}" if causes else ""),
        "production": identity,
        "warnings": status.get("quality", {}).get("warnings", []),
        "checklist": [
            f"{'✅' if checklist.get('production_alias') else '⚠️'} Production alias observed",
            f"{'✅' if checklist.get('quality_gates') else '⚠️'} Quality gates passed",
        ],
    }


def render_mlops_status() -> None:
    st.subheader("MLOps status")
    ok, data = api_get("/mlops/status")
    if not ok or not isinstance(data, dict):
        st.warning(f"Status unavailable: {data}")
        return
    messages = status_messages(data)
    if data["readiness"].get("status") == "ready":
        st.success(messages["readiness"])
    else:
        st.warning(messages["readiness"])
    st.caption(f"Production: {messages['production']}")
    for warning in messages["warnings"]:
        st.warning(warning)
    for item in messages["checklist"]:
        st.write(item)
    st.caption(data["authority"])


def init_session() -> None:
    if "user" not in st.session_state:
        st.session_state.user = None


def build_authenticator():
    """Construye el objeto de streamlit-authenticator con credenciales de la API.

    La verificación de contraseñas (bcrypt), la sesión y la cookie las gestiona
    streamlit-authenticator; las credenciales se cargan desde la base de datos
    a través del endpoint ``/auth/st/credentials``.
    """
    import streamlit_authenticator as stauth  # import perezoso (solo en runtime)

    ok, data = api_get("/auth/st/credentials")
    credentials = {"usernames": {}}
    if ok and isinstance(data, dict):
        credentials = data.get("credentials", credentials)

    authenticator = stauth.Authenticate(
        credentials,
        "focusai_auth",                                              # cookie_name
        os.getenv("FOCUSAI_COOKIE_KEY", "focusai_signing_key"),     # firma de la cookie
        30,                                                          # cookie_expiry_days
    )
    return authenticator


def _do_login(authenticator) -> None:
    """Invoca el widget de login soportando distintas versiones de la librería."""
    try:
        authenticator.login(location="main")
    except TypeError:
        authenticator.login("Iniciar sesión", "main")


def _do_logout(authenticator) -> None:
    try:
        authenticator.logout("Cerrar sesión", "sidebar")
    except TypeError:
        authenticator.logout("Cerrar sesión", location="sidebar")


def render_register() -> None:
    """Formulario de registro que persiste el usuario vía la API (hash bcrypt)."""
    with st.expander("¿No tienes cuenta? Crea una aquí"):
        with st.form("register_form"):
            username = st.text_input("Usuario nuevo")
            email = st.text_input("Email")
            password = st.text_input("Contraseña nueva", type="password")
            password2 = st.text_input("Repite la contraseña", type="password")
            submitted = st.form_submit_button("Crear cuenta", use_container_width=True)
            if submitted:
                if not username or not email or not password:
                    st.warning("Completa todos los campos.")
                elif password != password2:
                    st.warning("Las contraseñas no coinciden.")
                else:
                    ok, data = api_post(
                        "/auth/st/register",
                        {"username": username, "email": email, "password": password},
                    )
                    if ok:
                        st.success("Cuenta creada. Ahora inicia sesión.")
                    else:
                        st.error(data)


def render_auth() -> None:
    """Mensajes de estado + registro. El widget de login se renderiza en main()."""
    status = st.session_state.get("authentication_status")
    if status is False:
        st.error("Usuario o contraseña incorrectos.")
    elif status is None:
        st.caption("Inicia sesión arriba o crea una cuenta nueva.")

    render_register()


def resolve_current_user() -> dict | None:
    """Obtiene el registro del usuario autenticado (id/username/email) desde la API."""
    username = st.session_state.get("username")
    if not username:
        return None
    cached = st.session_state.get("user")
    if cached and cached.get("username") == username:
        return cached
    ok, data = api_get(f"/auth/st/user/{username}")
    if ok and isinstance(data, dict):
        st.session_state.user = data
        return data
    return None


def render_dashboard(authenticator) -> None:
    user = resolve_current_user()
    st.sidebar.title("FocusAI")
    if user:
        st.sidebar.write(f"Hola, **{user['username']}**")
    _do_logout(authenticator)
    if not user:
        st.warning("No se pudo cargar tu perfil desde la API. Vuelve a iniciar sesión.")
        return

    st.title("Diario de productividad")
    st.write("Escribe cómo fue tu día y el modelo Ensemble te clasificará.")
    render_mlops_status()

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

    if not (ok and isinstance(data, dict) and data.get("series")):
        st.info("Aún no hay historial. Clasifica tu primera entrada.")
        return

    df = pd.DataFrame(data["series"])
    df["dia"] = pd.to_datetime(df["dia"])

    # --- Filtro por rango de fechas ---
    min_day = df["dia"].min().date()
    max_day = df["dia"].max().date()
    date_range = st.date_input(
        "Filtrar por rango de fechas",
        value=(min_day, max_day),
        min_value=min_day,
        max_value=max_day,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
    else:
        start, end = min_day, max_day

    mask = (df["dia"].dt.date >= start) & (df["dia"].dt.date <= end)
    df = df.loc[mask].copy()
    if df.empty:
        st.info("No hay entradas en el rango seleccionado.")
        return

    df_plot = df.copy()
    df_plot["dia"] = df_plot["dia"].dt.strftime("%Y-%m-%d")
    fig = px.area(
        df_plot,
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

    st.download_button(
        "⬇️ Exportar serie (CSV)",
        data=df_plot.to_csv(index=False).encode("utf-8"),
        file_name="serie_procrastinacion.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if ok_d and isinstance(diarios, dict) and diarios.get("diarios"):
        st.subheader("Últimas entradas")
        hist = pd.DataFrame(diarios["diarios"])
        hist["created_at_dt"] = pd.to_datetime(hist["created_at"], errors="coerce")
        hist = hist[
            (hist["created_at_dt"].dt.date >= start) & (hist["created_at_dt"].dt.date <= end)
        ]
        show = hist[["created_at", "prediccion", "probabilidad", "texto"]].sort_values(
            "created_at", ascending=False
        )
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Exportar entradas (CSV)",
            data=show.to_csv(index=False).encode("utf-8"),
            file_name="diarios.csv",
            mime="text/csv",
            use_container_width=True,
        )


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

    authenticator = build_authenticator()

    if not st.session_state.get("authentication_status"):
        st.title("FocusAI")
        st.caption("Clasifica tus entradas de diario: Productivo vs Procrastinación")

    # Hidrata la sesión desde la cookie y renderiza el formulario de login si hace falta.
    _do_login(authenticator)

    if st.session_state.get("authentication_status"):
        render_dashboard(authenticator)
    else:
        render_auth()


if __name__ == "__main__":
    main()
