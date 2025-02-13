# get base image
FROM ubuntu:latest

# set environment variables
ENV user=ubuntu
ENV DEBIAN_FRONTEND=noninteractive
# set python version
ARG PYTHON_VERSION="3.12"
ENV PYTHON_VERSION=${PYTHON_VERSION}

# install required software and programmes for development environment
RUN apt-get update 
RUN apt-get install -y apt-utils vim curl wget unzip htop

# added cv2 dependencies
RUN apt-get install -y ffmpeg libsm6 libxext6

# set up home environment
RUN mkdir -p /home/${user} && chown -R ${user}: /home/${user}

# copy cat-classifier repo
COPY . /home/ubuntu/CatClassifier

# make data directory
RUN mkdir /home/${user}/CatClassifier/data
RUN mkdir /home/${user}/CatClassifier/model/checkpoints

# add deadsnakes ppa
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
# install required python and create virtual environment
RUN apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv 
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN /opt/venv/bin/python3 -m pip install -v -r /home/ubuntu/CatClassifier/requirements.txt
RUN /opt/venv/bin/python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

WORKDIR /home/${user}
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]