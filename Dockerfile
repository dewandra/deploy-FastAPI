
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# FROM python:3.11-slim
FROM python:3.10-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . .

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

# Install production dependencies.
RUN pip install -r requirements.txt

EXPOSE 8080

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED==1

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.

# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "main:app"]
