# Use the official Python image for Python 3.10
FROM python:3.10.12

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file into the container at /app
COPY requirements.txt /app/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole application directory into the container at /app
COPY . /app/

# Expose the port that FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "application.api:app", "--host", "0.0.0.0", "--port", "8000"]



# Assumed structure
# application/
#     api.py
# source/
#     utils.py
#     ast.py
# requirements.txt
# .env
# Dockerfile


# docker build -t AssistantBot
# docker run -d --name my-container -p 8000:8000 --env-file .env AssistantBot


