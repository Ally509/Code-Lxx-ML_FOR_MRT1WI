
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
from imblearn.over_sampling import ADASYN
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score

def run(seed, X1, y1):

    
    # Datas Splitting1
    x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=seed)

    # Standardized processing
    ss = StandardScaler()
    x_train=ss.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    clf = ExtraTreesClassifier(random_state=42)
    rfe = RFE(clf, n_features_to_select=50, step=10)
    x_train_rfe = rfe.fit_transform(x_train, y_train)
    x_test_rfe = rfe.transform(x_test)


   # Resampling
    resampling = SMOTETomek(random_state=55)
    x_train_resampled, y_train_resampled = resampling.fit_resample(x_train_rfe, y_train)

    # Classifier
    clf = ExtraTreesClassifier(random_state=42)
    clf.fit(x_train_resampled, y_train_resampled)
    y_pred = clf.predict(x_test_rfe)
    y_prob = clf.predict_proba(x_test_rfe)[:, 1]

    # Metrics
    clf_test_score = accuracy_score(y_test, y_pred)
    M = confusion_matrix(y_test, y_pred)

    print("Accuracy: {}".format(clf_test_score))
    print("AUC: {}".format(roc_auc_score(y_test, y_prob)))
    print("Sensitivity:", metrics.recall_score(y_test, y_pred))
    print("Specificity: {}".format(M[0][0] / (M[0][0] + M[0][1])))
    Gmean = geometric_mean_score(y_test, y_pred)
    print("G-mean:", Gmean)

if __name__ == '__main__':
    seed = [13, 57, 69, 258, 666, 1314, 1999, 2018, 2687, 54632]

    for i in range(len(seed)):
        print("Random Seed:", seed[i])
        start_time = time.time()

        df = pd.read_excel("data.xlsx")
        X = df.drop(labels=['ID', 'SOURCE', 'FNCLCC grade'], axis=1).fillna(0)
        y = df['FNCLCC grade']
        X1 = np.array(X)
        y1 = np.array(y)

        end_time = time.time()
        data_processing_time = end_time - start_time
        print("Data Processing Time:", data_processing_time)

        start_time = time.time()
        run(seed[i], X1, y1)
        end_time = time.time()
        running_time = end_time - start_time
        print("Running Time:", running_time)
        print("Total Time:", data_processing_time + running_time)
        print("--------------------------------")
