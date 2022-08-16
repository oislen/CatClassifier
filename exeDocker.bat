:: set docker settings
SET DOCKER_USER=oislen
SET DOCKER_REPO=catclass
SET DOCKER_TAG=latest
SET DOCKER_IMAGE=%DOCKER_USER%/%DOCKER_REPO%:%DOCKER_TAG%

:: remove existing docker containers and images
docker container prune -f
docker rm -f %DOCKER_IMAGE%

:: build docker image
call docker build --no-cache -t %DOCKER_IMAGE% . 
::call docker build -t %DOCKER_IMAGE% .

:: push docker container to DockerHub
:: call docker push %DOCKER_IMAGE%

:: run docker container
SET UBUNTU_DIR=/home/ubuntu
call docker run --shm-size=512m -p 8889:8888 -it %DOCKER_IMAGE%
:: docker images
:: docker ps -a
:: docker exec -it container_id /bin/bash