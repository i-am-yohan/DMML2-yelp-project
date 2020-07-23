#!/bin/bash


hdfs dfs -mkdir /Temp
hdfs dfs -copyFromLocal $1 /Temp
hdfs dfs -copyFromLocal $2 /Temp

python3 Review_Subset.py $1 $2

hdfs dfs -get /Temp/Output $3
