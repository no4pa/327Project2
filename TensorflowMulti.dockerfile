# Use a Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install TensorFlow and its dependencies
RUN pip install tensorflow tensorflow-datasets

# Copy your Python script into the container
COPY app/TensorflowMultiWorker.py .

# Set the entry point to your Python script
CMD ["python", "TensorflowMultiWorker.py"]
