# Use an official Python runtime as the base image
FROM amd64/python:3.9-buster AS project_env

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools \
    && pip install -r requirements.txt && sudo apt-get update && sudo apt-get install -y build-essential \
    && nvcc -c worker.cu -o worker.o \
    && g++ -shared -o worker.so worker.o -L/usr/local/cuda/lib64 -lcudart

FROM project_env

COPY . /app/

# Set the entrypoint command
CMD ["gunicorn", "--conf", "/app/gunicorn_conf.py", "main:app"]
