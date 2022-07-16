
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from imblearn.metrics import geometric_mean_score
import math
import time
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics

def run(seed, X1, y1):

    # Standardized processing
    ss = StandardScaler()
    std_data = ss.fit(X1)
    X_sm_std = std_data.transform(X1)

    clf = ExtraTreesClassifier(random_state=42)
    clf = RFE(clf, n_features_to_select=50, step=10).fit(X_sm_std, y1)
    X_rfe = clf.fit_transform(X_sm_std, y1)



    # #  Datas Splitting1 - split the data set first and then resampling
    x_train, x_test, y_train, y_test = train_test_split(X_rfe, y1, test_size=0.2, random_state=seed)
    print("the number of x_train:",y_train.shape)
    print("the number of x_test:", y_test.shape)

    ratio1 = sum(y_train==0)/sum(y_train==1)
    print('In Training set,train_0:train_1={:}:{:}={:.2f}'.format(sum(y_train==0),sum(y_train==1),ratio1))

    ratio2 = sum(y_test==0)/sum(y_test==1)
    print('In Testing set,test_0:tes_1={:}:{:}={:.2f}'.format(sum(y_test==0),sum(y_test==1),ratio2))


    resampling = SMOTE(random_state=55)
    x_train, y_train = resampling.fit_resample(x_train, y_train)
    #
    # y_train_1 = sum(y_train==1)
    # print("the sum of the 1 in training set",y_train_1)
    # y_train_0 = sum(y_train==0)
    # print("the sum of the 0 in training set",y_train_0)

    # # Datas Splitting2
    # x_train, x_test, y_train, y_test = train_test_split(X_rfe, y1, test_size=0.2, random_state=seed)
    #
    # resampling = SMOTE(random_state=0)
    # x_over, y_over = resampling.fit_resample(X_rfe, y1)
    #
    # x_train, x_test1, y_train, y_test1 = train_test_split(x_over, y_over, train_size=0.7, random_state=seed)
    #


    # Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_train, y_train)

    clf_test_score = clf.score(x_test, y_test)
    M = confusion_matrix(y_test, clf.predict(x_test))

    print("Accuracy: {}".format(clf_test_score))
    print("AUC: {}".format(roc_auc_score(y_test, clf.predict(x_test))))
    print("Sensitivity:", metrics.recall_score(y_test, clf.predict(x_test)))
    print("Specificity：{}".format(M[0][0] / (M[0][0] + M[0][1])))

    Gmean = geometric_mean_score(y_test, clf.predict(x_test))
    print("G-mean：",Gmean)



if __name__ == '__main__':

    seed = [13, 57, 69, 258, 666, 1314, 1999, 2018, 2687, 54632]
    all_time = []

    for i in range(len(seed)):
        print("random_seed：{}".format(seed[i]))
        fff_start = time.time()
        df = pd.read_excel("data.xlsx")
        X = df.drop(labels=['ID', 'SOURCE', 'FNCLCC grade'], axis=1).fillna(0)
        y = df['FNCLCC grade']
        X1 = np.array(X)
        y1 = np.array(y)
        fff_end = time.time()
        frist_time = fff_end - fff_start
        print("Data processing time：{}".format(frist_time))
        start_time = time.time()
        run(seed[i], X1, y1)
        end_time = time.time()
        second_time = end_time-start_time
        print("Running time：{}".format(second_time))
        print("Total time：{}".format(frist_time+second_time))
        print("--------------------------------")
        all_time.append(frist_time+second_time)


