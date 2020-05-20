#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import Window
from pyspark.sql.functions import col, expr, rank

import argparse 

def main(spark,model_use,path_assess,num_recommend) :

    test = spark.read.parquet(path_assess)  

    model_use = ALSModel.load(model_use)

    preds = model_use.transform(test) 

    windowSpec = Window.partitionBy('user_id').orderBy(col('prediction').desc())
    preds_for_RMSE = preds.select('user_id','book_id','rating','prediction',rank().over(windowSpec).alias('rank'))\
        .where('rank <= {}'.format(num_recommend))
    
    evaluator=RegressionEvaluator(metricName='rmse',
        labelCol="rating",
        predictionCol="prediction")

    rmse_all_data = evaluator.evaluate(preds)

    rmse_evaluation=evaluator.evaluate(preds_for_RMSE)

    perUserPredictedItemsDF = preds.select('user_id','book_id','prediction',rank().over(windowSpec).alias('rank'))\
        .where('rank <= {}'.format(num_recommend))\
        .groupBy('user_id')\
        .agg(expr('collect_list(book_id) as items'))

    perUserRelevantItemsDF = test.where(test.rating >= 3)\
        .groupBy('user_id')\
        .agg(expr('collect_list(book_id) as items'))

    perUserItemsRDD = perUserPredictedItemsDF.join(perUserRelevantItemsDF, 'user_id')\
        .rdd.map(lambda row: (row[1], row[2]))

    rankingMetrics = RankingMetrics(perUserItemsRDD)

    print('RMSE (ALL DATA) : ',rmse_all_data)
    print('Showing results for top {} predicted recommendations'.format(num_recommend))
    print('RMSE : ',rmse_evaluation)
    print('MAP : ',rankingMetrics.meanAveragePrecision)

if __name__ == '__main__' :

    # Create spark session
    spark_session = SparkSession.builder.appName('train')\
        .master('yarn').config('spark.executor.memory', '10g')\
        .config('spark.driver.memory', '10g').getOrCreate()

    ap = argparse.ArgumentParser()

    ap.add_argument("-a","--Model_File",required=True,
        help='Model file path to assess',
        type=str)
    ap.add_argument("-b", "--Assess_df_path", required=True,
        help="path to dataframe to evaluate",
        type=str)
    ap.add_argument("-c", "--num_recommend", required=True,
        help="number of ranked recommendations to evaluate",
        type=int)

    args = vars(ap.parse_args())

    print(args)

    main(spark_session,model_use=args['Model_File'],
        path_assess=args['Assess_df_path'],
        num_recommend=args['num_recommend'])