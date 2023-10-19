# This Dockerfile creates an image based on the official pyspark-notebook image
# Original: https://hub.docker.com/r/jupyter/pyspark-notebook
FROM jupyter/pyspark-notebook:2023-06-26

# Copy all files within 01_dev_requirements to build image
COPY /. . 

# Use pip3 to install all python requirements listed in requirements.txt
RUN pip3 install -r requirements.txt

ENV DEBIAN_FRONTEND noninteractive
RUN sudo apt-get install -y wget
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \ 
    && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list
RUN sudo apt-get update && sudo apt-get -y install google-chrome-stable
