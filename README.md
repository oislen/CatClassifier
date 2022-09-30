# Cat Classification

## Overview

This git repository contains code and configurations for implementing a Convolutional Neural Network to classify images containing cats or dogs. The data was sourced from the [dogs-vs-cats](https://www.kaggle.com/competitions/dogs-vs-cats/overview) Kaggle competition, and also from [freeimages.com](https://www.freeimages.com/) using a web scraper. Docker containers were used to deploy the application on an EC2 spot instances in order to scale up hardware and computation power. 

## Repo Contents

1. [aws](https://github.com/oislen/cat_classifier/tree/main/aws)
2. [conda](https://github.com/oislen/cat_classifier/tree/main/conda)
3. [data_prep](https://github.com/oislen/cat_classifier/tree/main/data_prep)
4. [kaggle](https://github.com/oislen/cat_classifier/tree/main/kaggle)
5. [model](https://github.com/oislen/cat_classifier/tree/main/model)
6. [ref](https://github.com/oislen/cat_classifier/tree/main/ref)
7. [webscrapers](https://github.com/oislen/cat_classifier/tree/main/webscrapers)

* The __aws__ subdirectory contains batch and shell scripts for configuring ec2 spot instances and the deploying docker container remotely. 
* The __conda__ subdirectory contains batch and shell scripts for creating a local conda environment for the project. 
* The __data_prep__ subdirectory contains python utility scripts to data cleansing and processing for modelling.
* The __kaggle__ subdirectory contains python scripts for downloading and unzipping competition data from Kaggle.
* The __model__ subdirectory contains python scripts for initiating and training CNN models.
* The ref subdirectory contains previous analysis and kernals on dogs vs cats classification from Kaggle community members.
* The __webscrapers__ subdirectory contains webscraping tools for downloading cats and dogs images from [freeimages.com](https://www.freeimages.com/).

## Application Scripts

The main dog and cat image classification application is contained within the root scripts:

1. [01_prg_kaggle_data.py](https://github.com/oislen/cat_classifier/tree/main/01_prg_kaggle_data.py)
2. [02_prg_scrape_imgs.py](https://github.com/oislen/cat_classifier/tree/main/02_prg_scrape_imgs.py)
3. [03_prg_keras_model.py](https://github.com/oislen/cat_classifier/tree/main/03_prg_keras_model.py)
4. [cons.py](https://github.com/oislen/cat_classifier/tree/main/cons.py)
5. [Dockerfile](https://github.com/oislen/cat_classifier/tree/main/Dockerfile)
6. [exeDocker.bat](https://github.com/oislen/cat_classifier/tree/main/exeDocker.bat)
7. [requirements.txt](https://github.com/oislen/cat_classifier/tree/main/requirements.txt)

* The __01_prg_kaggle_data.py__ script downloads / unzips the cat vs dogs competition data.
* The __02_prg_scrape_imgs.py__ script scrapes additional cat and dog images from [freeimages.com](https://www.freeimages.com/).
* The  __03_prg_keras_model.py__ script trains, fits and makes image predictions of the cat and dog images using a CNN model.
* The __cons.py__ script contains programme constants and configurations.
* The __Dockerfile__ builds the application container for deployment on ec2.
* The __exeDocker.bat__ executes the Docker build process locally on windows.
* The __requirements.txt__ file contains the python package dependencies for the application.

## Docker Container

The application docker container is available on dockerhub here:

https://hub.docker.com/repository/docker/oislen/cat-classifier