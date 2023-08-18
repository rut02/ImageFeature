# Use the official FastAPI image as the base image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Install the additional library
RUN apt-get update && apt-get install -y libglib2.0-0

# Copy the rest of the application code into the container
COPY . /app

# Expose port 8080
EXPOSE 8080

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
