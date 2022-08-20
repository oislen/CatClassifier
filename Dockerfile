# get base image
FROM ubuntu:20.04

# set environment variables
ENV user=ubuntu
ENV DEBIAN_FRONTEND=noninteractive

# install required software and programmes for development environment
RUN apt-get update 
RUN apt-get install -y apt-utils vim curl wget unzip git python3 python3-pip

# added cv2 dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# install required python packages
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
RUN playwright install

# set up home environment
RUN useradd ${user}
RUN mkdir -p /home/${user} && chown -R ${user}: /home/${user}

# clone cat-classifier git repo
RUN git clone https://github.com/oislen/cat_classifier.git /home/${user}/cat_classifier

# make data directory
RUN mkdir /home/${user}/cat_classifier/data
RUN mkdir /home/${user}/cat_classifier/scripts/checkpoints

# copy kaggle .zip file to data directory
COPY /home/${user}/cat_classifier/data/dogs-vs-cats.zip /home/${user}/cat_classifier/data

WORKDIR /home/${user}
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]