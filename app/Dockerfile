FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential\
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt


# Copy the current directory contents into the container at /app
COPY . /app


# Define environment variable for MLflow server if needed
#ENV MLFLOW_TRACKING_URI http://your_mlflow_server:5000

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
ENTRYPOINT ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8000"]
