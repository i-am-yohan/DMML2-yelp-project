#!/bin/bash


hdfs dfs -mkdir /Temp
hdfs dfs -copyFromLocal $1 /Temp
hdfs dfs -copyFromLocal $2 /Temp

wget https://raw.githubusercontent.com/i-am-yohan/DMML2-yelp-project/master/Spark-Subset/Review_Subset.py
python3 Review_Subset.py $1 $2
rm Review_Subset.py

hdfs dfs -get /Temp/Output Output
rm Output/_SUCCESS
mv Output/* $3

hdfs dfs -rm -r /Temp
