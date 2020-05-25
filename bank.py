# coding=utf-8
# @File  : bank.py
# @Author: 邱圆辉
# @Date  : 2020/5/16
# @Desc  : {银行精准营销解决方案}

import getopt
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

optimized_k = 7
train_file = './train_set.csv'

# Use which classifier to train the model
# 0: MyNaiveBayesClassifier
# 1: MyDecisionTreeClassifier
# 2: MyKNNClassifier
classifier = 0
select = False
f_num = 10


class BaseClassifier:
    def __init__(self, train_set_path, s):
        self.select = s
        self.train_set = pd.read_csv(train_set_path)
        self.x, self.y = self.data_preprocess()
        # split training set and testing set
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, test_size=0.06, random_state=2020)

    def data_preprocess(self):
        self.train_set.drop(['ID'], axis=1, inplace=True)
        self.train_set['pdays'].replace(-1, 999, inplace=True)

        # columns containing unknown values
        cols = ['job', 'education', 'contact', 'poutcome']
        # use each feature's mode as the default value
        default_val = ['blue-collar', 'secondary', 'cellular', 'failure']
        # replace all unknown value with default value of the corresponding feature
        for i, col in enumerate(cols):
            self.train_set[col].replace({'unknown': default_val[i]}, inplace=True)

        # convert category features to numerical features
        le = preprocessing.LabelEncoder()
        for col in self.train_set.columns:
            self.train_set[col] = le.fit_transform(self.train_set[col])

        y = self.train_set['y']
        self.train_set.drop(['y'], axis=1, inplace=True)
        if self.select:
            self.train_set = SelectKBest(chi2, k=f_num).fit_transform(self.train_set, y)
        else:
            self.train_set = self.train_set.to_numpy()

        # x = pd.get_dummies(data=x, columns=['housing', 'loan', 'month', 'job',
        #                                     'marital', 'education', 'contact', 'poutcome', 'default'])
        return self.train_set, y

    def show_features_distributions(self):
        self.train_set.hist(bins=50, figsize=(20, 15))
        plt.show()


class MyNaiveBayesClassifier(BaseClassifier):
    def __init__(self, train_set_path, s):
        BaseClassifier.__init__(self, train_set_path, s)
        self.clf = GaussianNB()
        self.clf_sigmoid = CalibratedClassifierCV(self.clf, cv=5)
        self.clf_sigmoid.fit(self.x_train, self.y_train)

    def predict(self):
        prediction = self.clf_sigmoid.predict(self.x_test)
        print(classification_report(self.y_test, prediction))

        y_predprob = self.clf_sigmoid.predict_proba(self.x_test)[:, 1]
        print("Accuracy: %.4g" % metrics.accuracy_score(self.y_test, prediction))
        print("AUC Score (Test): %f" % metrics.roc_auc_score(self.y_test, y_predprob))


class MyDecisionTreeClassifier(BaseClassifier):
    def __init__(self, train_set_path, s):
        BaseClassifier.__init__(self, train_set_path, s)
        self.clf = DecisionTreeClassifier(random_state=2020)
        self.clf.fit(self.x_train, self.y_train)

    def predict(self):
        prediction = self.clf.predict(self.x_test)
        print(classification_report(self.y_test, prediction))

        y_predprob = self.clf.predict_proba(self.x_test)[:, 1]
        print("Accuracy: %.4g" % metrics.accuracy_score(self.y_test, prediction))
        print("AUC Score (Test): %f" % metrics.roc_auc_score(self.y_test, y_predprob))


class MyKNNClassifier(BaseClassifier):
    def __init__(self, train_set_path, k, s):
        BaseClassifier.__init__(self, train_set_path, s)
        self.k = k  # categorize according to the nearest k neighbors

    def predict(self):
        # number of testing samples
        num_tests = self.x_test.shape[0]
        # (x_test - x_train) ^ 2 = x_test ^ 2 + x_train ^ 2 - 2 * x_test * x_train
        term1 = np.sum(np.square(self.x_test), axis=1, keepdims=True)
        term2 = np.sum(np.square(self.x_train), axis=1)
        term3 = -2 * np.dot(self.x_test, self.x_train.T)
        dist = np.sqrt(np.tile(term1, (1, term3.shape[1])) +
                       np.tile(term2, (term3.shape[0], 1)) + term3)

        y_pred = np.zeros(num_tests)
        # predict for each testing sample
        for i in range(num_tests):
            # position of the k nearest neighbors
            min_k_pos = np.argsort(dist[i])[:self.k]
            # category of the k nearest neighbors
            min_k_y = self.y_train.iloc[min_k_pos]
            y_pred[i] = np.argmax(np.bincount(min_k_y.tolist()))

        print(classification_report(self.y_test, y_pred))
        print('Accuracy: %.4g' % metrics.accuracy_score(self.y_test, y_pred))
        print("AUC Score (Test): %f" % metrics.roc_auc_score(self.y_test, y_pred))


def usage():
    print('Usage: {} -s/--select -f/--file <train dataset file> -c/--classifier <classifier type>'.format(sys.argv[0]))
    print('  -s, --select       select k-best features')
    print('  -f, --file         train dataset file path')
    print('  -c, --classifier   classifier type')
    print('      0 for naive bayes classifier')
    print('      1 for decision tree classifier')
    print('      2 for knn classifier')


def parse_args(argv):
    global select
    global train_file
    global classifier
    try:
        opts, args = getopt.getopt(argv, 'hsf:c:', ['help', 'select', 'file=', 'classifier='])
    except getopt.GetoptError:
        usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-s', '--select'):
            select = True
        elif opt in ('-f', '--file'):
            train_file = arg
        elif opt in ('-c', '--classifier'):
            if arg not in ('0', '1', '2'):
                usage()
                sys.exit(1)
            classifier = int(arg)


def main():
    global train_file
    global classifier
    parse_args(sys.argv[1:])
    if classifier == 0:
        print("Using MyNaiveBayesClassifier to train the model...")
        MyNaiveBayesClassifier(train_file, select).predict()
    elif classifier == 1:
        print("Using MyDecisionTreeClassifier to train the model...")
        MyDecisionTreeClassifier(train_file, select).predict()
    elif classifier == 2:
        print("Using MyKNNClassifier to train the model...")
        MyKNNClassifier(train_file, optimized_k, select).predict()


if __name__ == '__main__':
    main()
