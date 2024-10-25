# get base image
FROM ubuntu:latest

# set environment variables
ENV user=ubuntu
ENV DEBIAN_FRONTEND=noninteractive
# set git branch for cloning
ARG GIT_BRANCH
ENV GIT_BRANCH=${GIT_BRANCH}

# install required software and programmes for development environment
RUN apt-get update 
RUN apt-get install -y apt-utils vim curl wget unzip git

# added cv2 dependencies
RUN apt-get install -y ffmpeg libsm6 libxext6

# set up home environment
RUN mkdir -p /home/${user} && chown -R ${user}: /home/${user}

# clone cat-classifier git repo
RUN git clone https://github.com/oislen/CatClassifier.git --branch ${GIT_BRANCH} /home/${user}/CatClassifier

# make data directory
RUN mkdir /home/${user}/CatClassifier/data
RUN mkdir /home/${user}/CatClassifier/model/checkpoints

# install required python packages
RUN apt-get install -y python3.12 python3.12-venv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN /opt/venv/bin/python3.12 -m pip install -v -r /home/${user}/CatClassifier/requirements.txt

WORKDIR /home/${user}
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]