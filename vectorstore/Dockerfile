FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY app/ .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure writable permissions for MLflow tracking directory
RUN mkdir -p /tmp/mlruns && chmod -R 777 /tmp/mlruns

# Expose port
EXPOSE 8080

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
