#!/bin/bash

# Clone the GitHub repository
sudo apt-get install -y unzip

# get the dataset
mkdir data
cd data
wget https://huggingface.co/datasets/wanderkid/UniMER_Dataset/resolve/main/UniMER-Test.zip
wget https://huggingface.co/datasets/wanderkid/UniMER_Dataset/resolve/main/UniMER-1M.zip
unzip UniMER-Test.zip &
unzip UniMER-1M.zip
cd ..

# install pip
sudo apt update
sudo apt install python3-pip

# install requirements
pip install -r requirements.txt
pip install torchtext torcheval
