import numpy as np
import lightfm
from lightfm import data,evaluation
from sklearn import preprocessing
from scipy.sparse import coo_matrix 
import os
import time
import argparse

def main(data_path,original_downsample_amount,percentages,num_run,relevance_cutoff,k) :

    to_fill_time_train = np.zeros((len(percentages),num_run))
    to_fill_time_eval = np.zeros((len(percentages),num_run))
    to_fill_eval = np.zeros((len(percentages),num_run))

    chunk_read = np.array([np.genfromtxt(data_path+file,delimiter=',',dtype=int) for file in os.listdir(data_path) if file.endswith('.csv')])

    read_out = chunk_read[0]
    for arr in chunk_read[1:] :
        read_out = np.concatenate((read_out,arr))

    unique_users = set(read_out[:,0])

    for i,percent in enumerate(percentages) :

        sampled_users = np.random.choice(list(unique_users),int(len(unique_users)*float(percent)/original_downsample_amount),replace=False)

        sampled_data = read_out[np.where(np.isin(read_out[:,0],sampled_users))]

        sampled_unique_users = set(sampled_data[:,0])

        shuffled = np.random.choice(list(sampled_unique_users),len(sampled_unique_users),replace=False)
        train = sampled_data[np.where(np.isin(sampled_data[:,0],shuffled[:int(len(shuffled)*.8)]))]
        test = sampled_data[np.where(np.isin(sampled_data[:,0],shuffled[int(len(shuffled)*.8):]))]

        dict_enum = {}
        for j in test[:,0] :
            if j in dict_enum :
                dict_enum[j] += 1
            else :
                dict_enum[j] = 1
                
        dict_take = {}
        inds_to_train = []
        inds_to_test = []
        for j,v in enumerate(test) : 
            if v[0] in dict_take :
                if dict_take[v[0]] > int(dict_enum[v[0]]/2) :
                    inds_to_train.append(j)
                else :
                    inds_to_test.append(j)
                dict_take[v[0]] += 1
            else :
                dict_take[v[0]] = 1

        train = np.concatenate((train,test[inds_to_train]))
        test = test[inds_to_test]


        train_formatted,test_formatted = _informed_train_test(train_df=train,test_df=test)

        for n in range(num_run) :

            start = time.time()

            model=lightfm.LightFM()
            model.fit(train_formatted,epochs=10)

            model_fit_time = time.time() - start

            start_eval = time.time()

            to_predict = np.argwhere(test_formatted.tocsr() >  0)
            predictions = model.predict(to_predict[:,0],to_predict[:,1])

            MAP = []
            for user in np.unique(to_predict[:,0]) :
                add = _Average_Precision(test_formatted,to_predict,predictions,user,relevance_cutoff,k)
                MAP.append(add[0])

            eval_time = time.time() - start_eval

            to_fill_eval[i][n] = np.sum(MAP)/len(MAP)
            to_fill_time_eval[i][n] = eval_time
            to_fill_time_train[i][n] =  model_fit_time
            print(i,n,model_fit_time,eval_time)

    print(to_fill_time_train)
    print(to_fill_time_eval)
    print(to_fill_eval)


def _informed_train_test(train_df,test_df):
    
    test_df_use = test_df[np.isin(test_df[:,1],train_df[:,1])]
    
    trans_cat_train = dict()
    trans_cat_test = dict()
    for i in [0,1]:
        cate_enc = preprocessing.LabelEncoder()
        trans_cat_train[i] = cate_enc.fit_transform(train_df[:,i])
        trans_cat_test[i] = cate_enc.transform(test_df_use[:,i],)
    # --- Encode ratings:
    cate_enc = preprocessing.LabelEncoder()
    ratings = dict()
    ratings['train'] = cate_enc.fit_transform(train_df[:,3])
    ratings['test'] = cate_enc.transform(test_df_use[:,3])
    n_users = len(set(trans_cat_train[0]))
    n_items = len(set(trans_cat_train[1]))
    train = coo_matrix((ratings['train'], (trans_cat_train[0],trans_cat_train[1])),
                       shape=(n_users, n_items))
    test = coo_matrix((ratings['test'], (trans_cat_test[0],trans_cat_test[1])),
                      shape=(n_users, n_items))
    return train, test

def _Average_Precision(test_formatted,to_predict,predictions,user,relevance_cutoff,k) :
    inds = np.where(to_predict[:,0] == user)
    user_predictions = predictions[inds]
    user_actual = test_formatted.toarray()[to_predict[inds][:,0],to_predict[inds][:,1]]
    relevant_books = to_predict[inds][:,1][np.where(user_actual>= relevance_cutoff)[0]]
    
    if len(relevant_books) == 0 :
        return 1,0
    
    books_predicted_sort = to_predict[inds][:,1][np.array(list(reversed(user_predictions.argsort())))]
    
    books_predicted_sort = books_predicted_sort[:min(k,len(books_predicted_sort))]
    
    p = 0
    num_relevant = 0
    for i in range(len(books_predicted_sort)) :
        if books_predicted_sort[i] in relevant_books :
            num_relevant += 1
            p += num_relevant/(i+1)
            
    return p/num_relevant , len(books_predicted_sort)

if __name__ == '__main__' :

    ap = argparse.ArgumentParser()

    ap.add_argument("-a","--Data_Path",required=True,
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
    ap.add_argument("-e", "--Relevance_Cutoff", required=False,
        help="rating cutoff to be considered a relevant document >=",
        default=3.0,type=float)
    ap.add_argument("-f", "--k", required=True,
        help="number of ranked recommendations to evaluate",
        type=int)

    args = vars(ap.parse_args())

    print(args)

    main(data_path=args['Data_Path'],original_downsample_amount=args['Original_Downsample'],
        percentages=args['Downsampled_percents'],num_run=args['num_run'],
        relevance_cutoff=args['Relevance_Cutoff'],k=args['k'])
