# Use Python base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run the Flask application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
