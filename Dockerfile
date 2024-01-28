# Start by pulling a base image with Python 3.11.3
FROM laudio/pyodbc:2.1.0

# Copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# Switch working directory
WORKDIR /app

# Install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# Copy every content from the local file to the image
COPY . /app

# Expose port 8080
EXPOSE 8080

# Configure the container to run in an executed manner
CMD ["python", "app.py"]
