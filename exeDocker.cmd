:: set docker settings
SET DOCKER_USER=oislen
SET DOCKER_REPO=cat-classifier
SET DOCKER_TAG=latest
SET DOCKER_IMAGE=%DOCKER_USER%/%DOCKER_REPO%:%DOCKER_TAG%
SET DOCKER_CONTAINER_NAME=cc

:: remove existing docker containers and images
docker image rm -f %DOCKER_IMAGE%

:: build docker image
call docker build --no-cache -t %DOCKER_IMAGE% .

:: run docker container
call docker run --name %DOCKER_CONTAINER_NAME% --shm-size=512m --publish 8888:8888 --volume E:\GitHub\CatClassifier\.creds:/home/ubuntu/CatClassifier/.creds  --volume E:\GitHub\CatClassifier\report:/home/ubuntu/CatClassifier/report -it %DOCKER_IMAGE%
::call docker run --entrypoint sh --name %DOCKER_CONTAINER_NAME% ---shm-size=512m --publish 8888:8888 --volume E:\GitHub\CatClassifier\.creds:/home/ubuntu/CatClassifier/.creds  --volume E:\GitHub\CatClassifier\report:/home/ubuntu/CatClassifier/report -it %DOCKER_IMAGE%
::call docker run -it --entrypoint bash --name cc --shm-size=512m --volume /home/ec2-user/.creds:/home/ubuntu/CatClassifier/.creds --rm  oislen/cat-classifier:latest

:: useful docker commands
:: docker images
:: docker ps -a
:: docker exec -it {container_hash} /bin/bash
:: docker stop {container_hash}
:: docker start -ai {container_hash}
:: docker rm {container_hash}
:: docker image rm {docker_image}
:: docker push {docker_image}
