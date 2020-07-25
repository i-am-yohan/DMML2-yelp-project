from pyspark.sql import SQLContext, Window
from pyspark import SparkConf, SparkContext
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'remove line breaks from a CSV file'
    )

    parser.add_argument(
        'in_reviews_file',
        type=str,
        help='The input reviews file'
    )

    parser.add_argument(
        'in_id_file',
        type=str,
        help='The input Business IDs file'
    )

    args = parser.parse_args()

    in_csv_file1 = args.in_reviews_file
    in_id_file1 = args.in_id_file

    in_csv_file2 = in_csv_file1.split('/')
    in_id_file2 = in_id_file1.split('/')
    
    in_csv_file3 = in_csv_file2[len(in_csv_file2) - 1]
    in_id_file3 = in_id_file2[len(in_id_file2) - 1]

    conf = SparkConf().setMaster("local")
    sc = SparkContext(conf = conf)

    sqlContext = SQLContext(sc)

    In_review = sqlContext.read.csv("/Temp/{}".format(in_csv_file3) , header=True, inferSchema=True)
    In_Subset = sqlContext.read.csv("/Temp/{}".format(in_id_file3) , header=False, inferSchema=True)

    In_review.createOrReplaceTempView("In_review")
    In_Subset.createOrReplaceTempView("In_Subset")

    Out_DF = sqlContext.sql("""select bse.*
                            from In_review as bse
                            inner join In_Subset as sub on trim(upper(bse.business_id)) = trim(upper(sub._c0))
                        """)


    #Out_DF.createOrReplaceTempView("Out_DF")
    #sqlContext.sql("""select count(*) from Out_DF""").show()
    #sqlContext.sql("""select count(*) from In_review""").show()

    Out_DF.coalesce(1).write.mode('overwrite').csv('/Temp/Output' , header=True, quote="\u0000")
    
    sc.stop()
