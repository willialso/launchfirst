FROM ubuntu:22.04

WORKDIR /app

ENV PYTHONPATH="/app"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libboost-all-dev \
    swig \
    libgomp1 \
    libpython3-dev \
    g++-11 \
    git \
    wget \
    curl \
    python3.10 \
    python3-pip \
    poppler-utils \
    libglib2.0-0 \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 || true

# Set modern compiler
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100

# Install PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# **Install GCS client & auth libraries**
RUN pip install --no-cache-dir \
    google-cloud-storage \
    google-auth

# ✅ Download VADER lexicon for sentiment scoring
RUN python -m nltk.downloader vader_lexicon

# ✅ Install SciSpacy and base NLP model
RUN python -m spacy download en_core_web_sm && \
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# ✅ Copy local code (including scripts/, models/, main.py, etc)
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# ✅ Set default launch to FastAPI backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
