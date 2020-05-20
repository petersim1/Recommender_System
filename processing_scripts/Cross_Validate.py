from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from itertools import product
import pandas as pd
import numpy as np
import time

def Create_Grid(rank=[10],regParam=[0.1],maxIter=[10]) :

    return dict(zip(['rank','regParam','maxIter'],[rank,regParam,maxIter]))

def Split_Data(spark,df_to_split,df_to_add,percent_train=.6,partition_value=200) :

        '''l
        Input 
            - df_to_split = train/val dataset to split
            - df_to_add = test set subset to add back to training set
            - base_col = column to base the splits off of
            - percent_train = percent of base_col to put into training set
        Returns
            - train_split = training split
            - val_split = validation split
        '''
        
        percent_use = 2*percent_train/(percent_train+1)

        train_users,val_users = df_to_split.select('user_id')\
            .distinct().randomSplit([percent_use,1-percent_use])

        train = df_to_split.join(train_users,'user_id','inner')
        val_split = df_to_split.join(val_users,'user_id','inner')

        val_split.createOrReplaceTempView('val')

        val_split = spark.sql('SELECT *, ROW_NUMBER() \
            OVER (PARTITION BY user_id ORDER BY book_id DESC) AS pr \
            FROM val \
            ORDER BY user_id,pr')

        val_to_train = val_split.filter(val_split.pr%2 == 1).drop('pr')
        val = val_split.filter(val_split.pr%2 == 0).drop('pr')

        train = train.union(val_to_train).union(df_to_add)

        df_to_split.unpersist()
        df_to_add.unpersist()

        train = train.sort('user_id','book_id')
        train = train.repartition(partition_value,'user_id')
        val = val.sort('user_id','book_id')
        val = val.repartition(partition_value,'user_id')

        return train,val


def Cross_Validation(spark,train_val,test_to_val,hyperparam_grid,n=5,evaluation_metric='rmse',percent_train=.6,partition_value=200,random_seed=2020) :

    param_search = list(product(*hyperparam_grid.values()))

    to_fill = np.zeros((len(param_search),n+len(param_search[0])))

    evaluator=RegressionEvaluator(metricName=evaluation_metric,
            labelCol="rating",
            predictionCol="prediction")


    for i in range(n) :

        spark.catalog.clearCache()
        
        train_use,val_use = Split_Data(spark,train_val,test_to_val,partition_value=partition_value)

        #train_use.cache()
        #val_use.cache()
        
        count = 0
        for item in param_search :
            to_fill[count][0] = item[0]
            to_fill[count][1] = item[1]
            to_fill[count][2] = item[2]

            als = ALS(seed=random_seed,userCol='user_id',itemCol='book_id',ratingCol='rating'
                    ,coldStartStrategy='drop',rank=item[0],regParam=item[1],maxIter=item[2])

            model = als.fit(train_use)
            preds = model.transform(val_use)

            evaluation=evaluator.evaluate(preds)
            
            to_fill[count][i+len(param_search[0])] = evaluation
            count += 1

            print(i,count,item,evaluation)
            
            preds.unpersist()
        
            
        print()

    to_fill = pd.DataFrame(to_fill,columns=list(hyperparam_grid.keys()) + ['CV_'+str(i+1) for i in range(n)])

    return to_fill
