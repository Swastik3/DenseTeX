#!/bin/bash

# Clone the GitHub repository
git clone https://github.com/Swastik3/BuildSpaceResearch.git
sudo apt-get install -y unzip

cd BuildSpaceResearch

mkdir data
cd data
wget https://huggingface.co/datasets/wanderkid/UniMER_Dataset/resolve/main/UniMER-Test.zip
wget https://huggingface.co/datasets/wanderkid/UniMER_Dataset/resolve/main/UniMER-1M.zip
unzip UniMER-Test.zip &
unzip UniMER-1M.zip

cd ..

pip install -r requirements.txt




