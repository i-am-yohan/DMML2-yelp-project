# Yelp Data Anlaysis
Some work I did for a project where we tried to model the yelp review stars. The following shows the user how to execute the code correctly. The project is currently a work in progress.

## Subset to Restaraunts Only
The subfolder "Spark-Subsets" is executed as follows:
```
Execute_Review_Subset.sh <input file> <Business IDs> <Output File>
```
Ensure the HDFS is up and running prior to execution.

## Stratified Sampling
This program executes the stratified sampling on a specified number of reviews. It takes and even sample of each star rating and uses only the most recent reviews. The code also removes reviews written in a language other than English.
The whole proccess is executed using a bash script as follows:
```
Exec_Stratified_Sampling.sh <input csv> <output csv> <number of reviews to sample>
```
Ensure the HDFS is up and running before execution. This project takes about 8 hours to run in total.
