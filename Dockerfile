# Use python:3.8-slim as the base image
FROM python:3.8-slim

# Copy requirements.txt and install dependencies
COPY requirements.txt /drones_project/requirements.txt
RUN pip install -r /drones_project/requirements.txt

# Copy the project files into the container
COPY . /drones_project

# Set the working directory to /drones_project
WORKDIR /drones_project

# Define the entry point for the container
ENTRYPOINT ["python", "src/main.py"]
