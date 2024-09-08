# TensorFlow GPU image (Python 3.6)
FROM tensorflow/tensorflow:2.6.0-gpu

# Set the working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install necessary packages
RUN pip install --upgrade pip  # Upgrade pip
RUN pip install -r requirements.txt

# Check the installed version of scikit-learn
RUN python -c "import sklearn; print(sklearn.__version__)"