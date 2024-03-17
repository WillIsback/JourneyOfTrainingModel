#!/bin/bash

# Create the DataSet directory if it doesn't exist
mkdir -p ../DataSet
cd ../DataSet

# Download the datasets
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip the datasets
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip annotations_trainval2017.zip

# Clean up the zip files
rm annotations_trainval2017.zip
rm train2017.zip
rm val2017.zip
rm test2017.zip

cd ..