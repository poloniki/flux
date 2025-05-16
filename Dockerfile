FROM nvidia/cuda:12.4.1-base-ubuntu22.04
# Install dependencies
RUN apt-get update -y && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6

# Set working directory
WORKDIR /app

# Copy the current directory (sd-scripts) to the container
COPY . /app/

# Install sd-scripts dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

# Install Torch, Torchvision, and Torchaudio for CUDA 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Create directory structure
RUN mkdir -p /app/models/unet \
    /app/models/clip \
    /app/models/vae \
    /app/datasets \
    /app/outputs

# Define volumes to persist data across container lifecycles
VOLUME ["/app/models", "/app/outputs", "/app/datasets"]

# Set environment variables
ENV PYTHONPATH=/app
RUN pip install --upgrade --force-reinstall triton==2.1.0

# Make train_model.py executable
RUN chmod +x /app/train_model.py

# Set the entrypoint to run the script
ENTRYPOINT ["python3", "/app/train_model.py"]

# Default CMD arguments that can be overridden
CMD ["--help"]