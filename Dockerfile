# Get current python image
#   (use -bullseye variants on local arm64/Apple Silicon)
FROM python:3.13.0-bullseye

# Update image
RUN apt-get update && apt-get upgrade -y

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Install deflicker locally in editable mode
RUN pip3 install -e .

# Copy project code into the image
COPY . .
