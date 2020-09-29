# multi stage build
# FROM ubuntu:16.04
# FROM alpine:3.8
# FROM python:3.7.3
FROM python_req

ENV PYTHONUNBUFFERED 1
run pip install --upgrade pip
RUN apt-get update
# RUN add-apt-repository ppa:mc3man/trusty-media
# RUN apt-get update
# RUN apt-get dist-upgrade
RUN echo "Y" | apt-get install ffmpeg

# source directory for the application
RUN mkdir /code

# setup woking directory
WORKDIR /code

# copy all the files of the project into the container from local machine
COPY . /code/

# run pip install for installing the requrements
RUN pip install -r requirements.txt

WORKDIR /code/djangorestui
EXPOSE 8000

# CMD ["python", "manage.py", "runserver"]


