import argparse
import math

from pyspark.sql import SQLContext, Window
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import row_number, udf, col
from datetime import datetime
from langdetect import detect


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Sample an equal number of reviews from the review CSV file'
    )

    parser.add_argument(
        'in_reviews_file',
        type=str,
        help='The input reviews file'
    )

    parser.add_argument(
        'nobs',
        type=int,
        help='The number of observations'
    )

    args = parser.parse_args()

    in_csv_file1 = args.in_reviews_file
    nobs = args.nobs

    in_csv_file2 = in_csv_file1.split('/')
    in_csv_file3 = in_csv_file2[len(in_csv_file2) - 1]

    conf = SparkConf().setMaster("local")
    sc = SparkContext(conf = conf)
    sqlContext = SQLContext(sc)

    In_Reviews = sqlContext.read.csv("/Temp/{}".format(in_csv_file3)
                                     , sep = '|'
                                     , quote = "'"
                                     , escape = '\\'
                                     , header = True
                                     )

    def Str_2_Dte(x):
        return(datetime.strptime(x , '%Y-%m-%d %H:%M:%S'))

    Str_2_Dte_udf = udf(Str_2_Dte)
    In_Reviews = In_Reviews.withColumn('dttm',Str_2_Dte_udf(col('date')))



    #Filter out non-English reviews
    def l_det(x):
        try:
            output = detect(x)
        except:
            output = 'undetected'
        return output


    l_det_udf = udf(l_det)
    In_Reviews = In_Reviews.withColumn('lang', l_det_udf(col('text')))
    In_Reviews = In_Reviews.filter("lang = 'en'")

    In_Reviews.createOrReplaceTempView("In_Reviews")
    In_Reviews = sqlContext.sql("""
    select * from In_Reviews
    where lang = 'en'
    order by dttm desc
    """)
    In_Reviews = In_Reviews.drop('lang')
    In_Reviews = In_Reviews.drop('date')

    Star_5 = In_Reviews.filter("stars = '5.0'").limit(math.ceil(nobs/5))
    Star_4 = In_Reviews.filter("stars = '4.0'").limit(math.ceil(nobs/5))
    Star_3 = In_Reviews.filter("stars = '3.0'").limit(math.ceil(nobs/5))
    Star_2 = In_Reviews.filter("stars = '2.0'").limit(math.ceil(nobs/5))
    Star_1 = In_Reviews.filter("stars = '1.0'").limit(math.ceil(nobs/5))

    Out_DF = Star_1.union(Star_2)
    Out_DF = Out_DF.union(Star_3)
    Out_DF = Out_DF.union(Star_4)
    Out_DF = Out_DF.union(Star_5)

    Out_DF.coalesce(1).write.mode('overwrite').csv('/Temp/Output' , header=True , escape="\\", quote="'", encoding = 'UTF-8',sep="|")
    sc.stop()