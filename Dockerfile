# Stage 1: Build Python environment
FROM python:3.9 AS python_env

# Install Python packages
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools \
    && pip install -r requirements.txt

# Stage 2: Build CUDA environment
FROM nvidia/cuda:12.4.0-base-ubuntu22.04 AS cuda_env

# Install CUDA tools
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cuda-command-line-tools-12-1 \
    cuda-nvcc-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy Python packages from the Python stage
COPY --from=python_env /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=python_env /usr/local/bin /usr/local/bin

# Copy source files
COPY . /app/

# Compile and link CUDA code
RUN nvcc -c worker.cu -o worker.o \
    && g++ -shared -o worker.so worker.o -L/usr/local/cuda/lib64 -lcudart

# Set the entrypoint command
CMD ["gunicorn", "--conf", "/app/gunicorn_conf.py", "main:app"]
