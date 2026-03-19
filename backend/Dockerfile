FROM python:3.11-slim

WORKDIR /app

COPY requirements_render.txt .
RUN pip install --no-cache-dir -r requirements_render.txt

COPY main.py .
ENV PORT=8000
EXPOSE 8000

CMD ["python", "main.py"]
