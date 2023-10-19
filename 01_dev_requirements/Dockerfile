# This Dockerfile creates an image based on the official pyspark-notebook image
# Original: https://hub.docker.com/r/jupyter/pyspark-notebook
FROM jupyter/pyspark-notebook:2023-06-26

# Copy all files within 01_dev_requirements to build image
COPY /. . 

# Use pip3 to install all python requirements listed in requirements.txt
RUN pip3 install -r requirements.txt

# USER root
# ENV DEBIAN_FRONTEND noninteractive
# RUN apt-get install -y wget
# RUN apt-get update && apt-get install -y gnupg
# RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \ 
#     && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list
# RUN apt-get update && apt-get -y install google-chrome-stable

# USER root
# RUN apt-get update && apt-get install -y gnupg
# # install google chrome
# RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
# RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
# RUN apt-get -y update
# RUN apt-get install -y google-chrome-stable
# # install chromedriver
# RUN apt-get install -yqq unzip
# RUN apt-get install -yqq curl
# RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
# RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/