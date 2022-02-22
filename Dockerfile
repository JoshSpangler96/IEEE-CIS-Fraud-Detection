# Dockerfile

# Use latest Python runtime as a parent image
FROM python:3.8-slim-buster

# Meta-data
LABEL maintainer="Joshua Spangler <jdspangelr96@gmail.com>" \
      description="Machine Learning Application for the \
      IEEE-CIS Fraud Detection Kaggle Competition."

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Download required package for LightGBM
RUN apt-get update -y && apt-get install -y apt-utils
RUN apt-get install -y libgomp1

# pip install
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install --user -r /app/predict_api/requirements.txt

# Make port available to the world outside this container
EXPOSE 1080

# Create mountpoint
VOLUME /app/data

# ENTRYPOINT allows us to specify the default executible
ENTRYPOINT ["python"]

# CMD sets default arguments to executable which may be overwritten when using docker run
CMD ["/app/predict_api/app.py"]