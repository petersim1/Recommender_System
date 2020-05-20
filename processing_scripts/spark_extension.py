#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS,ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import Window
from pyspark.sql.functions import col, expr,rank

import argparse 
import time
import numpy as np

# load custom module.
import data_preparation

def main(spark,data_file,original_downsample_amount,percentages,num_run,k) :

    data_prep = data_preparation.Prep_Data(spark)

    df = spark.read.parquet(data_file)  

    to_fill_time_train = np.zeros((len(percentages),num_run))
    to_fill_time_eval = np.zeros((len(percentages),num_run))
    to_fill_eval = np.zeros((len(percentages),num_run))

    windowSpec_predicted = Window.partitionBy('user_id').orderBy(col('prediction').desc())

    for i,percent in enumerate(percentages) :

        # downsample accordingly. Correct for when the original input file is already downsampled.
        df_subbed = data_prep.DownSample(df,float(percent)/original_downsample_amount) 

        perUserRelevantItemsDF = df_subbed.where(df_subbed.rating >= 3)\
            .groupBy('user_id')\
            .agg(expr('collect_list(book_id) as items'))

        for n in range(num_run) :

            start = time.time()
            als = ALS(seed=2020,userCol='user_id',itemCol='book_id',ratingCol='rating',
                coldStartStrategy='drop') #use default hyperparameters.

            model = als.fit(df_subbed)
            model_fit_time = time.time() - start

            preds = model.transform(df_subbed)

            start_eval = time.time()

            perUserPredictedItemsDF = preds.select('user_id','book_id','prediction',rank().over(windowSpec_predicted).alias('rank'))\
                .where('rank <= {}'.format(k))\
                .groupBy('user_id')\
                .agg(expr('collect_list(book_id) as items'))

            perUserItemsRDD = perUserPredictedItemsDF.join(perUserRelevantItemsDF, 'user_id')\
                .rdd.map(lambda row: (row[1], row[2]))

            rankingMetrics = RankingMetrics(perUserItemsRDD)

            MAP_value = rankingMetrics.meanAveragePrecision
            eval_time = time.time() - start_eval

            to_fill_eval[i][n] = MAP_value
            to_fill_time_eval[i][n] = eval_time
            to_fill_time_train[i][n] =  model_fit_time
            print(i,n,model_fit_time,eval_time,MAP_value)

    print(to_fill_time_train)
    print(to_fill_time_eval)
    print(to_fill_eval)

if __name__ == '__main__' :

    # Create spark session
    spark_session = SparkSession.builder.appName('extenstion')\
        .master('yarn').config('spark.executor.memory', '10g')\
        .config('spark.driver.memory', '10g').getOrCreate()

    ap = argparse.ArgumentParser()

    ap.add_argument("-a","--Data_File",required=True,
        help='Path to interactions data',
        type=str)
    ap.add_argument("-b","--Original_Downsample",required=False,
        help='if file used is originally downsampled, input value',
        default=1.0,type=float)
    ap.add_argument("-c", "--Downsampled_percents", required=True,
        help="downsampled percents to measure",
        nargs='+', type=float)
    ap.add_argument("-d", "--num_run", required=False,
        help="number of runs per downsample percent",
        default=10,type=int)
    ap.add_argument("-e", "--k", required=True,
        help="number of ranked recommendations to evaluate",
        type=int)

    args = vars(ap.parse_args())

    print(args)

    main(spark_session,data_file=args['Data_File'],original_downsample_amount=args['Original_Downsample'],
        percentages=args['Downsampled_percents'],num_run=args['num_run'],k=args['k'])