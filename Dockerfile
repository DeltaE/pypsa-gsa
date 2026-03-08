FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the console
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies
# We copy just the requirements.txt first to leverage Docker cache
COPY dashboard/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the dashboard directory
COPY dashboard /app/dashboard

# Set the working directory to dashboard so absolute imports like `components...` work
WORKDIR /app/dashboard

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# CMD ["sh", "-c", "exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:server"]
CMD ["sh", "-c", "exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 app:server"]
