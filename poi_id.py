#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from tester import dump_classifier_and_data
import pprint
import pandas as pd
import test

## change all the other column into numeric type
def change_to_num(df,col_name):
    df[col_name] = pd.to_numeric(df[col_name], errors = 'coerce')

def load_data_in_pd_format():
	with open("final_project_dataset.pkl", "r") as data_file:
	    data_dict = pickle.load(data_file)
	    # remove outlier
	    data_dict.pop("TOTAL")
	    data_dict.pop('LOCKHART EUGENE E') 
	    df =pd.DataFrame(data_dict)
	    df = df.T
	    # remove non-numerical columns in df
	    df.drop('email_address',1, inplace=True)

	    for name in list(df.columns):
	    	change_to_num(df,name)
		df.fillna(value=0, inplace=True)
	    return df 


def create_bonus_salary_ratio(row):
	if row['bonus'] == 0 or row['salary'] ==0:
		return 0
	else:
		return row['bonus']/row['salary']

def get_features_and_labels(df,features_list):	
	labels = df['poi']	
	index_of_poi = features_list.index("poi")
	features_list.pop(index_of_poi)
	features = df[features_list]
	return (features,labels)

def turn_df_to_dict(df):
	data_dict = df.to_dict(orient="index")
	return data_dict

from sklearn import cross_validation
from sklearn import preprocessing
def train_test_feature_label_split(features, labels, scale=False):
    features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features, labels, test_size=0.3, random_state=1)   
    if scale:
        min_max_scaler = preprocessing.MinMaxScaler()
        feature_train = min_max_scaler.fit_transform(features_train)
        feature_test = min_max_scaler.fit_transform(features_test)    
    return (features_train, features_test, labels_train, labels_test)


from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

def get_feature_importance(features, labels,scale = False):    
    importance_list = []
    min_max_scaler = preprocessing.MinMaxScaler()
    sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.5, random_state=0)
    sss.get_n_splits(features, labels)   
    for train_index, test_index in sss.split(features, labels):        
        labels_train, labels_test = labels.ix[train_index], labels.ix[test_index]
        if not scale: 
            features_train, features_test = features.ix[train_index,:], features.ix[test_index,:]
        else:
            features_train, features_test = features.ix[train_index,:], features.ix[test_index,:]
            features_train = min_max_scaler.fit_transform(features_train)
            features_test = min_max_scaler.fit_transform(features_test) 

        # fit an Extra Trees model to the data
        clf = ExtraTreesClassifier()
        clf.fit(features_train,labels_train)
        # display the relative importance of each attribute
        importance = clf.feature_importances_
        features_list = list(features.columns)
        importance_dict=dict(zip(features_list,importance))
        importance_list.append(importance_dict)        
    df = pd.DataFrame(importance_list)
    
    ordered_dict =sorted(df.mean().to_dict().items(),key=lambda t: -t[1])
    order_dict_df = pd.DataFrame(ordered_dict)

    pprint.pprint(order_dict_df)

    return order_dict_df   

def get_top_13_feature_list(order_dict_df,df):
	top13=order_dict_df.ix[:,0][:13]
	list_of_feature_frames = []
	for i in range(1,14):
	    list_of_feature_frames.append(df[top13[:i]])
	return list_of_feature_frames


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
def get_number_of_features(list_of_feature_frames, labels):
	clf = GaussianNB()
	precision_recall_list =[]
	for i in range(13):
		precision_recall = test.test_classifier(list_of_feature_frames[i], labels, clf, scaled= False)
		precision_recall_list.append(precision_recall)

	precision_recall_df = pd.DataFrame(precision_recall_list)

	index_names = []
	for i in range(13):
	    index_names.append("top_{}".format(i+1))
	precision_recall_df.index = index_names

	pprint.pprint(precision_recall_df)
	precision_recall_df['pr_sum'] = precision_recall_df.precision+precision_recall_df.recall
	index_ = list(precision_recall_df.pr_sum).index(precision_recall_df.pr_sum.max(axis=0))
	top_count = precision_recall_df.index[index_]

	return top_count

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
def try_naive_bayes(selected_features, labels, scale = False):
	clf = GaussianNB()
	if not scale:
		test.test_classifier(selected_features, labels, clf, scaled= False)
	else:
		test.test_classifier(selected_features, labels, clf, scaled= True)
	return clf 

from sklearn.neighbors.nearest_centroid import NearestCentroid
def try_nearest_centroid(selected_features, labels, scale = False):
	clf = NearestCentroid(shrink_threshold=0.05)
	if not scale:
		test.test_classifier(selected_features, labels, clf, scaled= False)
	else:
		test.test_classifier(selected_features, labels, clf, scaled= True)
	return clf

def main():
	df = load_data_in_pd_format()
	df['bonus_salary_ratio'] = df.apply(create_bonus_salary_ratio, axis=1)
	my_dataset = turn_df_to_dict(df)

	features_list = list(df.columns)	
	features, labels = get_features_and_labels(df,features_list)
	
	#focus on not_scaled, as will use naive bayes
	feature_importance_dataframe = get_feature_importance(features, labels,scale = False)
	#get_feature_importance(features, labels,scale = True)
	list_of_feature_frames = get_top_13_feature_list(feature_importance_dataframe,df)

	top_count= get_number_of_features(list_of_feature_frames, labels)
     
    # from the top_count, we know that we should use the top 3 features, 
    # so use the top three features in the feature_importance_dataframe
    
	selected_features = ['deferred_income',
	                     'exercised_stock_options',
	                     'total_stock_value',
	                     ]
	features= df[selected_features ]
	print "this is the naive bayes without engineered feature"
	try_naive_bayes(features, labels, scale = False)
	print "this is the nearest_centroid without engineered feature"
	try_nearest_centroid(features, labels, scale = True)

    # see how naive bayes  and nearest centroid work when the feature I engineered is not included
	selected_features_with_ratio = ['deferred_income',
	                                'exercised_stock_options',
	                                'total_stock_value',	                                
                                    'bonus_salary_ratio']
	features= df[selected_features_with_ratio ]
	print "this is the naive bayes with engineered feature"
	try_naive_bayes(features, labels, scale = False)

	print "this is the nearest_centroid with engineered feature"
	try_nearest_centroid(features, labels, scale = True)

	
	features_list =['poi',
	                'deferred_income',
	                'exercised_stock_options',
	                'total_stock_value',
	                ]

	features= df[selected_features ]
	clf=try_naive_bayes(features, labels, scale = False)
	
	dump_classifier_and_data(clf, my_dataset, features_list)

if __name__ == '__main__':
    main()
