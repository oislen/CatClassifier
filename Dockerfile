# get base image
FROM python:3.12-slim@sha256:2fe5997d249a808b8eeea52c58a1dbffbba28754dc11699ef5c029f2d818ce79

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
#RUN apt-get install -y imagemagick=8:7.1.1.43+dfsg1-1+deb13u5
#RUN apt-get install -y libssl-dev=3.5.4-1~deb13u2
#RUN apt-get install -y libpq-dev=17.8-0+deb13u1
#RUN apt-get install -y libpng-dev=1.6.48-1+deb13u3 libpng16-16t64=1.6.48-1+deb13u3
#RUN apt-get install -y linux-libc-dev=6.12.73-1

# added cv2 dependencies
RUN apt-get install -y ffmpeg libsm6 libxext6

# set up home environment
RUN adduser ${user}
RUN mkdir -p /home/${user} && chown -R ${user}: /home/${user}

# copy cat-classifier repo
COPY --chown=${user}:${user} . /home/${user}/CatClassifier
RUN ls -R /home/${user}/CatClassifier/data
# set working directory
WORKDIR /home/${user}/CatClassifier

# make data directory
#RUN mkdir -p /home/${user}/CatClassifier/data
#RUN mkdir -p /home/${user}/CatClassifier/data/checkpoints

# install required python packages
COPY --from=ghcr.io/astral-sh/uv@sha256:68224c7eb575bc13e4723f8831331db68555125fac3939fdde78cdb6000667ad /uv /uvx /bin/
RUN uv sync

EXPOSE 8888
EXPOSE 5000

#ENTRYPOINT ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
ENTRYPOINT ["uv", "run", "flask", "--app", "model/api.py", "run", "--host", "0.0.0.0", "--port", "5000"]