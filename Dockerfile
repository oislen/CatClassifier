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

## install anaconda
#RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
#RUN shasum -a 256 ~/Anaconda3-2024.10-1-Linux-x86_64.sh
#RUN bash ~/Anaconda3-2024.10-1-Linux-x86_64.sh -b -p /home/ubuntu/anaconda3
#RUN export PATH="/home/ubuntu/anaconda3/bin:PATH"
#RUN conda init
#RUN source ~/.bashrc
# RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia -y

# add deadsnakes ppa
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
# install required python and create virtual environment
RUN apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv 
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN /opt/venv/bin/python3 -m pip install -v -r /home/ubuntu/CatClassifier/requirements.txt

WORKDIR /home/${user}
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]