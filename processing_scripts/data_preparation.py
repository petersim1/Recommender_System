import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

import sys
import random
import os
import subprocess

class Prep_Data() :
    
    def __init__(self,spark,partition_value=200,random_seed=None) :

        '''
        Purpose:
            This module will perform the necessary pre-processing of a parquet file.
            Module contains functions to Trim low frequency user (remove users with less than # of books),
            and will remove is_read=False items if desired.
            Downsamples (when testing, this is useful) based off user_id.
            Applies indexing to user/item columns, since ALS requires numeric input values.
            Also performs splits of data into training/testing/validation sets.

        Inputs:
            - spark = spark session.
            - interactions_read = path to csv/parquet interactions file (if csv, will convert to parquet then read in/out)
            - users_read = path to csv/parquet users file (if csv, will convert to parquet then read in/out)
            - books_read = path to csv/parquet books file (if csv, will convert to parquet then read in/out)
            - partition_value = desired partition value (can lower if you plan to downsample, or if data is small)

        Created module will contain the read in, sorted, repartitioned dataframe
        '''

        interactions_schema = 'user_id INT,book_id INT, is_read INT,rating INT, is_reviewed INT'
        users_schema = 'user_id_csv INT,user_id STRING'
        books_schema = 'book_id_csv INT,book_id STRING'

        if random_seed is None :
            self.seed = int(random.random()*100)
        else :
            self.seed = random_seed

        self.partition_value = partition_value
        self.spark=spark

        self.interactions_schema = interactions_schema
        self.users_schema = users_schema
        self.books_schema = books_schema

    def Get_Data_Local(self,interactions_read,users_read,books_read) :

        new_name = interactions_read.replace('csv','parquet')

        if os.path.exists(new_name) :
            interactions = self.spark.read.parquet(interactions_read.replace('csv','parquet'))
            users = self.spark.read.parquet(users_read.replace('csv','parquet'))
            books = self.spark.read.parquet(books_read.replace('csv','parquet'))
        else :
            if os.path.exists(interactions_read) :
                interactions = self.spark.read.csv(interactions_read,schema=self.interactions_schema)
                users = self.spark.read.csv(users_read,schema=self.users_schema)
                books = self.spark.read.csv(books_read,schema=self.books_schema)
                interactions.write.parquet(new_name)
                users.write.parquet(users_read.replace('csv','parquet'))
                books.write.parquet(books_read.replace('csv','parquet'))
                interactions = self.spark.read.parquet(new_name)
                users = self.spark.read.parquet(users_read.replace('csv','parquet'))
                books = self.spark.read.parquet(books_read.replace('csv','parquet'))
            else :
                raise 'file does not exist'
        
        interactions = interactions.sort('user_id','book_id')

        interactions = interactions.repartition(self.partition_value,'user_id')


        return interactions,users,books

    def Get_Data_HDFS(self,interactions_read,users_read,books_read) :

        new_name_interactions = interactions_read.replace('csv','parquet').split('/')[-1]
        new_name_users = users_read.replace('csv','parquet').split('/')[-1]
        new_name_books = books_read.replace('csv','parquet').split('/')[-1]
        
        val = subprocess.call('hdfs dfs -test -e {}'.format(new_name_interactions),shell=True)
        if val == 0 :
            interactions = self.spark.read.parquet(new_name_interactions)
        else :
            val = subprocess.call('hdfs dfs -test -e {}'.format(interactions_read),shell=True)
            if val == 0 :
                interactions = self.spark.read.csv(interactions_read,schema=self.interactions_schema)
                interactions.write.parquet(new_name_interactions)
                interactions = self.spark.read.parquet(new_name_interactions)
            else :
                raise 'file does not exist'
        
        val = subprocess.call('hdfs dfs -test -e {}'.format(new_name_users),shell=True)
        if val == 0 :
            users = self.spark.read.parquet(new_name_users)
        else :
            val = subprocess.call('hdfs dfs -test -e {}'.format(users_read),shell=True)
            if val == 0 :
                users = self.spark.read.csv(interactions_read,schema=self.users_schema)
                users.write.parquet(new_name_users)
                users = self.spark.read.parquet(new_name_users)
            else :
                raise 'file does not exist'
        
        val = subprocess.call('hdfs dfs -test -e {}'.format(new_name_books),shell=True)
        if val == 0 :
            books = self.spark.read.parquet(new_name_books)
        else :
            val = subprocess.call('hdfs dfs -test -e {}'.format(books_read),shell=True)
            if val == 0 :
                books = self.spark.read.csv(books_read,schema=self.books_schema)
                books.write.parquet(new_name_books)
                books = self.spark.read.parquet(new_name_books)
            else :
                raise 'file does not exist'
        
        interactions = interactions.sort('user_id','book_id')
        interactions = interactions.repartition(self.partition_value,'user_id')

        return interactions
        
    def Trim_LowNum(self,interactions,min_allowed=10,cut_not_read=True) :

        '''
        Trims out low occurring users, and can trim out items not yet read.

        Input :
            - df = dataframe to be trimmed
            - min_allowed = INT. required minimum # of books read for a user.
            - cut_not_read = True/False. If True, will cut out books not read (affects count of books/user)
        Output:
            - trimmed dataframe.
        '''
        interactions.cache()

        interactions.createOrReplaceTempView('df')
        
        if cut_not_read :
            file_trimmed = self.spark.sql('SELECT * FROM df \
WHERE (user_id IN \
(SELECT user_id as num_users \
FROM df \
WHERE is_read=1 \
GROUP BY user_id \
HAVING COUNT(*) >= {})) AND (is_read=1)'.format(min_allowed))
        else :
            file_trimmed = self.spark.sql('SELECT * FROM df \
WHERE user_id IN \
(SELECT user_id as num_users \
FROM df \
GROUP BY user_id \
HAVING COUNT(*) >= {})'.format(min_allowed))
            
        interactions.unpersist()
        
        file_trimmed.cache()

        file_trimmed = file_trimmed.sort('user_id','book_id')
        file_trimmed = file_trimmed.repartition(self.partition_value,'user_id')
            
        return file_trimmed
    
    def DownSample(self,interactions,percent_use) :
        
        '''
        Used for testing purposes, take a proportion to make testing faster

        Input
             - df : dataframe to be downsampled
             - base_col : reference column to downsample off of
             - percent_use : downsample ratio (0.01 takes 1% of original "base_col")
        Returns:
            sampled version based on input column
        '''
        sampled_df = interactions.select('user_id').distinct()\
            .sample(percent_use,seed=self.seed).join(interactions,'user_id','inner')
      
        interactions.unpersist()
            
        sampled_df = sampled_df.sort('user_id','book_id')
        sampled_df = sampled_df.repartition(self.partition_value,'user_id')

        sampled_df.cache()

        return sampled_df

    def Index_columns(self,df,*cols) :

        '''
        required for ALS. Takes integers as input, not strings.
        Input 
            - df : dataframe to perform indexing on.
            - *cols : column name arguments to be indexed.
        Returns 
            - indexed dataframe
            - indexing. Will be need to re-map users/book_ids
        '''
        
        indexer = [StringIndexer(inputCol=column, outputCol=column+"_numeric") for column in cols ]

        pipeline = Pipeline(stages=indexer)

        df = pipeline.fit(df).transform(df)
        
        return df,indexer

    def Create_TestSet(self,interactions,percent_train=.6) :

        '''
        Creates separate Test Set, and the test set subset that will be rejoined to the training set.
        Input
            - df = indexed dataframe to split.
            - base_col = column to base the splits off of
            - percent_train = percent of base_col to put into training set
        Returns 
            - train_val_set = training/val split to further split into validation and testing.
            - test_set = final testing set
            - test_to_val = test set subset to add back to training set.
        '''
        self.spark.catalog.clearCache()
        
        interactions.cache()
        
        percent_use = percent_train+(1-percent_train)/2

        train_users,test_users = interactions.select('user_id')\
            .distinct().randomSplit([percent_use,1-percent_use])
        train_val = interactions.join(train_users,'user_id','inner')
        test_split = interactions.join(test_users,'user_id','inner')

        interactions.unpersist()
        
        test_split.cache()
        train_val.cache()

        test_split.createOrReplaceTempView('test')
                        
        test_split = self.spark.sql('SELECT *, ROW_NUMBER() \
            OVER (PARTITION BY user_id ORDER BY book_id DESC) AS pr \
            FROM test \
            ORDER BY user_id,pr')

        test_set = test_split.filter(test_split.pr%2 == 1).drop('pr')
        test_to_val = test_split.filter(test_split.pr%2 == 0).drop('pr')

        test_set = test_set.sort('user_id','book_id')
        test_set = test_set.repartition(self.partition_value,'user_id')
        test_to_val = test_to_val.sort('user_id','book_id')
        test_to_val = test_to_val.repartition(self.partition_value,'user_id')
        train_val = train_val.sort('user_id','book_id')
        train_val = train_val.repartition(self.partition_value,'user_id')
        
        test_split.unpersist()

        print('Writing training/validation split...')
        train_val.write.parquet('train_val_split.parquet.tmp','overwrite')
        print('Writing test split...')
        test_set.write.parquet('test_set_split.parquet.tmp','overwrite')
        print('Writing test-to-training split...')
        test_to_val.write.parquet('test_to_val.parquet.tmp','overwrite')
        
        train_val  = self.spark.read.parquet('train_val_split.parquet.tmp')
        test_to_val = self.spark.read.parquet('test_to_val.parquet.tmp')

        return train_val,test_to_val