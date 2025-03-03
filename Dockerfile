FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MONGODB_HOST=mongodb
ENV MONGODB_PORT=27017

# Expose the port
EXPOSE 8000

# Command to run the server
CMD ["python", "server.py"]

