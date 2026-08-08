FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN pip install --upgrade pip && \
    pip install streamlit plotly requests pandas "streamlit-authenticator==0.3.2" PyYAML

COPY frontend /app/frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
