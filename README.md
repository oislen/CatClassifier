# Cat Classification

## Overview

This git repository contains code and configurations for implementing Convolutional Neural Networks to classify images containing cats or dogs. The data was sourced from [freeimages.com](https://www.freeimages.com/) using a variety of web scrapers, and also from the [dogs-vs-cats](https://www.kaggle.com/competitions/dogs-vs-cats/overview) Kaggle competition. Docker containers were used for deployment of code base on EC2 spot instances in order to scale up hardware and computation power. 

## Contents

1. [environments](https://github.com/oislen/cat_classifier/tree/main/environments)
2. [scripts](https://github.com/oislen/cat_classifier/tree/main/scripts)

The environments subdirectory contains batch and shell scripts for creating conda environments, configuring ec2 spot instances and deploying docker containers. The scripts subdirectory contains the main code base for data generation and modelling. Most notably data web scrapers, image processing utilities and Keras CNN models.