#!/bin/bash

#cd ~

hdfs dfs -mkdir /Temp
hdfs dfs -copyFromLocal $1 /Temp


python3 /home/hduser/DMML2/Project/ETL/Stratified_Sampling/Stratified.py $1 $3

hdfs dfs -get /Temp/Output Output
rm Output/_SUCCESS
mv Output/* $2

hdfs dfs -rm -r /Temp
