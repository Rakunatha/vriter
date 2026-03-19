FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone --depth=1 https://github.com/Rudrabha/Wav2Lip.git && \
    pip install --no-cache-dir -r Wav2Lip/requirements.txt 2>/dev/null || true

RUN git clone --depth=1 https://github.com/OpenTalker/SadTalker.git || true

COPY vrite/    vrite/
COPY ui/       ui/
COPY scripts/  scripts/
COPY run.py    .

RUN mkdir -p models/sadtalker uploads outputs tmp

RUN mkdir -p /root/.streamlit && \
    printf "[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\nmaxUploadSize = 500\n\n[theme]\nbase = \"dark\"\nprimaryColor = \"#6366f1\"\n\n[browser]\ngatherUsageStats = false\n" \
    > /root/.streamlit/config.toml

ENV PORT=8501
EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

CMD streamlit run ui/app.py \
    --server.address=0.0.0.0 \
    --server.port=${PORT} \
    --server.headless=true \
    --server.fileWatcherType=none
