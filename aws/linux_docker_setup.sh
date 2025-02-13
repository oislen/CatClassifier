# NOTE:
# 1. make sure to add ip address to security groups inbound rules
# 2. make sure to increase volume in /dev/nvme0n1 (/dev/xvda) e.g. 100gb

# linux file formatting
# sudo apt-get install -y dos2unix
# dos2unix ./linux_docker_setup.sh 

#-- EC2 Spot Instance Checks --#

# check available memory and cpu capacity
free -h
df -h
lscpu
# calculate percentage of used memory
free -m | awk 'FNR == 2 {print $3/($3+$4)*100}'
# check gpu status
nvidia-smi
# watch -n 0.5 nvidia-smi

#-- Configure Permissions and Overcommit Settings --#

# reset premission for the /opt /dev /run and /sys directories
ls -larth /.
sudo chmod -R 777 /opt /dev /run /sys/fs/cgroup
sudo chmod 775 /var/run/screen
ls -larth /.
# mask permissoin for home docker file
sudo chmod 700 ~/.creds
sudo chmod 600 ~/.creds/*
# update overcommit memory setting
cat /proc/sys/vm/overcommit_memory
echo 1 | sudo tee /proc/sys/vm/overcommit_memory

#-- Increase EBS Volume --#

# verify that the root partition mounted under "/" is full (100%)
df -h
# gather details about your attached block devices
lsblk
lsblk -f
# mount the temporary file system tmpfs to the /tmp mount point
sudo mount -o size=10M,rw,nodev,nosuid -t tmpfs tmpfs /tmp
# Run the growpart command to grow the size of the root partition or partition 1
sudo growpart /dev/nvme0n1 1 
# Run the lsblk command to verify that partition 1 is expanded
lsblk
# Expand the file system
sudo xfs_growfs -d /
# file system on partition 1 is expanded
sudo resize2fs /dev/nvme0n1p1
# use the df -h command to verify that the OS can see the additional space
df -h
# Run the unmount command to unmount the tmpfs file system.
sudo umount /tmp

#-- Download Required Programmes --#

# update os
sudo apt-get update -y
# install required base software
sudo apt-get install -y htop vim tmux dos2unix docker git
# remove unneed dependencies
sudo apt-get autoremove

#-- Pull Git Repo --#

# pull git repo
sudo mkdir /home/ubuntu
sudo git clone https://github.com/oislen/CatClassifier.git --branch main /home/ubuntu/CatClassifier
cd /home/ubuntu/CatClassifier
sudo git config --global --add safe.directory /home/ubuntu/CatClassifier
sudo mkdir /home/ubuntu/CatClassifier/.creds
sudo cp -r ~/.creds/* /home/ubuntu/CatClassifier/.creds/
sudo chmod 755 /home/ubuntu/CatClassifier/.creds
sudo chmod 600 /home/ubuntu/CatClassifier/.creds/*

#-- Pull and Run Docker Contianer --#

# apply doc2unix to docker_extract_data.sh script
dos2unix ~/docker_extract_data.sh
#set docker tag
docker_user='oislen'
docker_repo='cat-classifier'
docker_tag='latest'
docker_image=$docker_user/$docker_repo:$docker_tag
docker_container_name='cc'
# login to docker
sudo gpasswd -a $USER 
sudo systemctl start docker
sudo chmod 666 /var/run/docker.sock
#export PATH="$HOME/.docker/config.json:$PATH"
#echo '{"credsStore":"osxkeychain"}' > ~/.docker/config.json
#docker login --username oislen --password `cat ~/.creds/docker`
cat ~/.creds/docker | docker login --username oislen --password-stdin
# pull docker container
docker pull $docker_image
# run pulled docker container
#docker run --shm-size=512m -p 8889:8888 -it $docker_image
docker run --name $docker_container_name --shm-size=512m --publish 8888:8888 --volume /home/ubuntu/CatClassifier/.creds:/home/ubuntu/CatClassifier/.creds  --volume /home/ubuntu/CatClassifier/report:/home/ubuntu/CatClassifier/report --rm -it --entrypoint bash $docker_image
#docker run --shm-size=512m -p 8889:8888 -d $docker_image
#docker run -it -d <container_id_or_name> /bin/bash