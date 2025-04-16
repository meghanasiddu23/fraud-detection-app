# 1. Use an official Python runtime as base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy everything into the container
COPY . /app

# 4. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose port
EXPOSE 5000

# 6. Run the app
CMD ["python", "app.py"]
