from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

#this is modified from the tester.py

def test_classifier(features, labels, clf, scaled= False):
    min_max_scaler = preprocessing.MinMaxScaler()
    
    sss = StratifiedShuffleSplit(n_splits=300, test_size=0.5, random_state=0)
    sss.get_n_splits(features, labels)
    #print(sss)       

    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    
    print "dim of features are "   
    print features.shape
    
    for train_index, test_index in sss.split(features, labels):
        #print("TRAIN:", train_index, "TEST:", test_index)
        labels_train, labels_test = labels.ix[train_index], labels.ix[test_index]
        if not scaled: 
            features_train, features_test = features.ix[train_index,:], features.ix[test_index,:]
        else:
            features_train, features_test = features.ix[train_index,:], features.ix[test_index,:]
            features_train = min_max_scaler.fit_transform(features_train)
            features_test = min_max_scaler.fit_transform(features_test) 

    ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
#             print prediction
#             print truth
#             print "****************************"
            if prediction == False and truth == False:
                true_negatives += 1
            elif prediction == False and truth == True:
                false_negatives += 1
            elif prediction == True and truth == False:
                false_positives += 1
            elif prediction == True and truth == True:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    print "true_negatives count is {}".format(true_negatives)
    print "false_negatives count is {}".format(false_negatives)
    print "false_positives count is {}".format(false_positives)
    print "true_positives count is {}".format(true_positives)
    print "=========================================="
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)

        recall = 1.0*true_positives/(true_positives+false_negatives)
        precision_and_recall = {"precision":precision,"recall":recall}
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
        
        return precision_and_recall
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."
        print "++++++++++++++++++++++++++++++++++++++"
    return precision_and_recall
    
