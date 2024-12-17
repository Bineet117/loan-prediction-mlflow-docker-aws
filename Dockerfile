# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY src/requirements.txt ./requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --timeout=100 -r requirements.txt

# Copy the application code
COPY src/ ./src/

# Copy the model file to the container
COPY models/model.pkl ./src/models/

# Set the working directory for the application
WORKDIR /app/src

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
