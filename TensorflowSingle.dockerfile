# Use the official Python image as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Install the required libraries
RUN pip install tensorflow tensorflow-datasets

# Copy the Python script into the container
COPY app/TensorflowSingleWorker.py .

# Run the Python script when the container starts
CMD ["python", "TensorflowSingleWorker.py"]

