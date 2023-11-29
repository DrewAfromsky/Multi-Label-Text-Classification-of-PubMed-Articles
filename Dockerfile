# Start from the official Python base image.
FROM python:3.10 as base

# Set the current working directory to /code.
# This is where requirements.txt file and the app directory will be located
WORKDIR /code

# Copy the file with the requirements to the /code directory.
COPY ./requirements.txt /code/requirements.txt
# COPY ./tests /tests

# Install the package dependencies in the requirements file.
# The --no-cache-dir option tells pip to not save the downloaded packages locally.
# The --upgrade option tells pip to upgrade the packages if they are already installed.
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the ./main.py file inside the /code directory.
COPY ./main.py /code/

# Create a new target called test that will run the test suite
# FROM base as test
# CMD ["python3", "-m", "pytest", "-vv", "/tests/code/test_main.py"]