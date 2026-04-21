FROM node:20-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/backend/requirements.txt

COPY frontend/package.json frontend/package-lock.json /app/frontend/
RUN npm --prefix /app/frontend ci

COPY backend /app/backend
COPY configs /app/configs
COPY frontend /app/frontend
COPY hf_spaces/start.sh /app/start.sh

ENV NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
RUN npm --prefix /app/frontend run build

RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
