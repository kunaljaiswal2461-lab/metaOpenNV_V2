FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for TA-Lib or basic builds
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from root and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Aligns with openenv task risk_aware_trading (20-bar window → 203-dim vector).
ENV WINDOW_SIZE=20

# Expose the Hugging Face Space port
EXPOSE 7860

# Run the root entrypoint
CMD ["python", "app.py"]
