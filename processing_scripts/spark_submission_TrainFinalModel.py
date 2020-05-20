#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

import argparse 

def main(spark,train_val_file,test_to_val_file,test_file,rank,regParam,maxIter,file_out,\
    evaluation_metric='rmse',partition_value=200,random_seed=2020) :

    train_val = spark.read.parquet(train_val_file)
    test_to_val = spark.read.parquet(test_to_val_file)
    test = spark.read.parquet(test_file)
    
    als = ALS(seed=random_seed,userCol='user_id',itemCol='book_id',ratingCol='rating',
            coldStartStrategy='drop')

    train = train_val.union(test_to_val).sort('user_id','book_id').repartition(partition_value,'user_id')

    als = ALS(seed=random_seed,userCol='user_id',itemCol='book_id',ratingCol='rating',
        coldStartStrategy='drop',
        rank=rank,
        regParam=regParam,
        maxIter=maxIter)

    model = als.fit(train)
    preds = model.transform(test)
            
    evaluator=RegressionEvaluator(metricName=evaluation_metric,
        labelCol="rating",
        predictionCol="prediction")

    evaluation=evaluator.evaluate(preds)

    print('Final out-of-sample Model evaluation for {} : {}'.format(evaluation_metric,evaluation))

    model.write().overwrite().save(file_out)

if __name__ == '__main__' :

    # Create spark session
    spark_session = SparkSession.builder.appName('train')\
        .master('yarn').config('spark.executor.memory', '10g')\
        .config('spark.driver.memory', '10g').getOrCreate()

    ap = argparse.ArgumentParser()

    ap.add_argument("-a","--Model_File_write",required=True,
        help='Model file to write out to',
        type=str)
    ap.add_argument("-b", "--TrainVal_data_path", required=True,
        help="path to the training data",
        type=str)
    ap.add_argument("-c", "--test_to_val_data_path", required=True,
        help="path to the testing data to pull back into training split",
        type=str)
    ap.add_argument("-d", "--test_data_path", required=True,
        help="path to the testing data",
        type=str)
    ap.add_argument("-e","--rank",required=False,   
        help="List format, rank values to test in cross validation",
        default=10, type=int)
    ap.add_argument("-f","--regParam",required=False,   
        help="List format, regParam values to test in cross validation",
        default=0.1, type=float)
    ap.add_argument("-g","--maxIter",required=False,   
        help="List format, maxIter values to test in cross validation",
        default=10, type=int)
    args = vars(ap.parse_args())

    print(args)

    main(spark_session,train_val_file=args['TrainVal_data_path'],
        test_to_val_file=args['test_to_val_data_path'],
        test_file=args['test_data_path'],rank=args['rank'],regParam=args['regParam'],maxIter=args['maxIter'],
        file_out=args['Model_File_write'],evaluation_metric='rmse',partition_value=200,random_seed=2020)