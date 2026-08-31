FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-app.txt

COPY . .

EXPOSE 8501 8000

CMD ["sh", "-c", "streamlit run app/dashboard.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
