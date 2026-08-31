FROM python:3.10-slim

WORKDIR /app

ENV PORT=10000
ENV CORE_MODEL_GAZE_ENABLED=false
ENV CORE_MODEL_SAM_ENABLED=false
ENV CORE_MODEL_CLIP_ENABLED=false

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
