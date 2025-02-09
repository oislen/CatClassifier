# Cat Classification

## Overview

This git repository contains code and configurations for implementing a Convolutional Neural Network to classify images containing cats or dogs. The data was sourced from the [dogs-vs-cats](https://www.kaggle.com/competitions/dogs-vs-cats/overview) Kaggle competition, and also from [freeimages.com](https://www.freeimages.com/) using a web scraper. Docker containers were used to deploy the application on an EC2 spot instances in order to scale up hardware and computation power. 

## Analysis Results

See the analysis results notebook for a summary of the project; including image processing, CNN architecture and model performance.
* https://nbviewer.org/github/oislen/CatClassifier/blob/main/report/torch_analysis_results.ipynb

## Docker Container

The application docker container is available on dockerhub here:

https://hub.docker.com/repository/docker/oislen/cat-classifier
