#!/bin/bash

#cd ~

hdfs dfs -mkdir /Temp
hdfs dfs -copyFromLocal $1 /Temp

wget https://raw.githubusercontent.com/i-am-yohan/DMML2-yelp-project/master/Stratified_Sampling/Stratified.py
python3 Stratified.py $1 $3
rm Stratified.py

hdfs dfs -get /Temp/Output Output
rm Output/_SUCCESS
mv Output/* $2

hdfs dfs -rm -r /Temp
