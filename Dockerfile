FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# we install libpq-dev because psycopg needs it to talk to Postgres
RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev gcc \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc \
    && rm -rf /var/lib/apt/lists/*


COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]