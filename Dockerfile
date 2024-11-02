# Use a minimal base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies securely
COPY requirements.txt requirements.txt

# Use a non-root user for security
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY . .

# Set permissions for the app directory to the non-root user
USER appuser

# Expose the port Flask is running on
EXPOSE 5000

# Set environment to production
ENV FLASK_ENV=production

# Run the application
CMD ["python", "app.py"]
