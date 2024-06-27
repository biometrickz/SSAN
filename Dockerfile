# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# COPY req.txt /app/
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -U scikit-learn scipy matplotlib
RUN pip install --no-cache-dir -r req.txt

# Run solver.py when the container launches. Adjust the command according to actual usage.
# CMD ["python", "solver.py", "--data_dir", "/mnt/8TB/ml_projects_yeldar/", "--model_type", "SSAN_R", "--batch_size", "256", "--img_size", "112", "--protocol", "Patchnet", "--num_epochs", "1200"]
# CMD ["python", "solver.py", "--data_dir", "data_dir", "--model_type", "SSAN_R", "--batch_size", "256", "--img_size", "112", "--protocol", "Patchnet", "--num_epochs", "1200"]

