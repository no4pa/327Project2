# Use the official Python image as the base image
FROM pytorch/pytorch:latest

# Set the working directory inside the container
WORKDIR /app

COPY requirements.txt .

# Install the required libraries
RUN pip install -r requirements.txt

# Copy the Python script into the container
COPY app/PytorchThreeWorker.py .

# Run the Python script when the container starts
CMD ["python", "PytorchThreeWorker.py"]

