# This file wile be used to build the container. See a complete tutorial at:
# https://medium.com/@doedotdev/docker-flask-a-simple-tutorial-bbcb2f4110b5


# ------ Define the base image we will start from ------
FROM frolvlad/alpine-miniconda3

# Here, we choose a base image that includes conda
# In other use cases, we could have used alternatives such as:
# FROM python:3.7
# FROM ubuntu:latest


# ------ Set up the docker /app folder ------
WORKDIR /app
COPY ../ /app

# We copy the content of the parent folder into a specific docker dir.
# If this doesn't work, move all docker-related files to the parent folder.
# The current folder content a .dockerignore file with the list of dir to ignore.


# ------ Create the conda environment ------
RUN conda env create -f environment.yml -n docker_env
RUN echo "CHECKPOINT"

# Here, we install the requirements through a conda environment creation
# based on a .yml file. Alternatively, we could use a requirements.txt file:
# RUN pip install -r requirements.txt


# ------ Make RUN commands use the conda environment ------
SHELL ["conda", "run", "-n", "docker_env", "/bin/bash", "-c"]

# This section is just to make sure that the next commands will use the new env.


# ------ The code to run when the container is started ------
ENTRYPOINT ["conda", "run", "-n", "docker_env", "python", "solar_app.py"]
# CMD ["solar_app.py"] # alternative