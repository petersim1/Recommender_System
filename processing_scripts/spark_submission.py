#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pyspark.sql import SparkSession
# import custom modules and scripts
import data_preparation
import Cross_Validate

import argparse 
import subprocess
import pandas as pd

def main(spark,CV_file_out,Recreate_Split,partition_value,rank,regParam,maxIter,
    interactions_file,users_file,books_file,percent_downsample) :

    val = subprocess.call('hdfs dfs -test -e {}'.format('train_val_split_fulldata.parquet.tmp'),shell=True)
    val += subprocess.call('hdfs dfs -test -e {}'.format('test_to_val_fulldata.parquet.tmp'),shell=True)

    if (Recreate_Split): # Will consume time, but recreates files from scratch
        if (interactions_file is None) or (users_file is None) or (books_file is None) :
            raise ValueError('Need to input file paths, quitting...')
        data_prep = data_preparation.Prep_Data(spark,partition_value,random_seed=None)
        interactions = data_prep.Get_Data_HDFS(interactions_file,users_file,books_file)
        interactions = data_prep.Trim_LowNum(interactions,min_allowed=10,cut_not_read=True)
        if percent_downsample is not None :
            interactions = data_prep.DownSample(interactions,percent_use=percent_downsample)
        train_val,test_to_val = data_prep.Create_TestSet(interactions,percent_train=.6)

    if val > 0 :
        raise ValueError('Needed files do not exist, need to recreate (change to Recreate_Split=True). Quitting...')
    train_val = spark.read.parquet('train_val_split_fulldata.parquet.tmp')
    test_to_val = spark.read.parquet('test_to_val_fulldata.parquet.tmp')

    hyperparam_grid = Cross_Validate.Create_Grid(rank,regParam,maxIter)

    out = Cross_Validate.Cross_Validation(spark,train_val,test_to_val,hyperparam_grid,n=3,
        evaluation_metric='rmse',percent_train=.6,partition_value=partition_value,random_seed=2020)

    out.to_csv(CV_file_out,index=False)


if __name__ == '__main__' :

    # Create spark session
    spark_session = SparkSession.builder.appName('train')\
        .master('yarn').config('spark.executor.memory', '15g')\
        .config('spark.driver.memory', '15g').getOrCreate()

    ap = argparse.ArgumentParser()

    ap.add_argument("-a", "--CV_file_out", required=True,
        help="csv file to write out the pandas DF of CV performance.",
        type=str)
    ap.add_argument("-b", "--Recreate_Split", required=False,
        help="True/False , whether or not you want to re-trim/split data, will require additional inputs if True",
        default=False,type=bool)
    ap.add_argument("-c", "--partition_value", required=False,
        help="Input value to repartition files to",
        default=200,type=int)
    ap.add_argument("-d","--rank",required=False,   
        help="List format, rank values to test in cross validation",
        default=[10],nargs='+', type=int)
    ap.add_argument("-e","--regParam",required=False,   
        help="List format, regParam values to test in cross validation",
        default=[.1],nargs='+', type=float)
    ap.add_argument("-f","--maxIter",required=False,   
        help="List format, maxIter values to test in cross validation",
        default=[10],nargs='+', type=int)
    ap.add_argument("-g","--interactions_file",required=False,   
        help="path to hdfs interactions data",
        default=None,type=None)
    ap.add_argument("-i","--users_file",required=False,   
        help="path to hdfs users data",
        default=None,type=None)
    ap.add_argument("-j","--books_file",required=False,   
        help="path to hdfs books data",
        default=None,type=None)
    ap.add_argument("-k","--percent_downsample",required=False,   
        help="path to hdfs books data",
        default=.2,type=float)
    args = vars(ap.parse_args())

    print(args)

    main(spark_session, CV_file_out=args['CV_file_out'], Recreate_Split=args['Recreate_Split'],
        partition_value=args['partition_value'], 
        rank=args['rank'], regParam=args['regParam'], maxIter=args['maxIter'], 
        interactions_file=args['interactions_file'], users_file=args['users_file'], books_file=args['books_file'],
        percent_downsample=args['percent_downsample'])

    

    

