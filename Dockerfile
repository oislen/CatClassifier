# get base image
FROM python:3.12

# set environment variables
ENV user=ubuntu
ENV DEBIAN_FRONTEND=noninteractive
# set python version
ARG PYTHON_VERSION="3.12"
ENV PYTHON_VERSION=${PYTHON_VERSION}

# install required software and programmes for development environment
RUN apt-get update
RUN apt-get install -y apt-utils vim curl wget unzip tree htop adduser
# install trivy image vulnerability patches
RUN apt-get install -y imagemagick=8:7.1.1.43+dfsg1-1+deb13u5
RUN apt-get install -y libssl-dev=3.5.4-1~deb13u2
RUN apt-get install -y libpq-dev=17.8-0+deb13u1
RUN apt-get install -y libpng-dev=1.6.48-1+deb13u3 libpng16-16t64=1.6.48-1+deb13u3
RUN apt-get install -y linux-libc-dev=6.12.73-1

# added cv2 dependencies
RUN apt-get install -y ffmpeg libsm6 libxext6

# set up home environment
RUN adduser ${user}
RUN mkdir -p /home/${user} && chown -R ${user}: /home/${user}

# copy cat-classifier repo
COPY . /home/${user}/CatClassifier
# set working directory
WORKDIR /home/${user}/CatClassifier

# make data directory
RUN mkdir /home/${user}/CatClassifier/data
RUN mkdir /home/${user}/CatClassifier/model/checkpoints

# install required python packages
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv sync

EXPOSE 8888
ENTRYPOINT ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]