# Use python-slim for a smaller image footprint and lower security vulnerability surface
FROM python:3.11-slim

# Install system dependencies required for matplotlib and drawing rendering
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install dependencies (cached in layers)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files (leveraging .dockerignore)
COPY . .

# Create the outputs folder and grant full read/write permissions.
# Crucial for Hugging Face Spaces since the container runs under a non-root user (UID 1000)
RUN mkdir -p outputs && chmod 777 outputs

# Hugging Face Spaces default port
EXPOSE 7860

# Run Flask backend server
CMD ["python", "app.py"]