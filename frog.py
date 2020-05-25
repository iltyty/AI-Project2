# coding=utf-8
# @File  : frog.py
# @Author: 邱圆辉
# @Date  : 2020/5/17
# @Desc  : {青蛙叫声聚类分析}

import sys
import getopt
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import v_measure_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score

train_file = './Frogs_MFCCs.csv'

# Clustering type
# 0: MyAggClustering
# 1: MyKMeansClustering
clustering = 0
# number of frog families (clusters)
n_clusters = 4
# whether to select k-best features from original train dataset
select = False
# number of features after selecting
f_num = 12


class BaseClustering:
    def __init__(self, train_set_path, k, s):
        self.k = k
        self.select = s
        self.train_set = pd.read_csv(train_set_path)
        self.y = self.train_set['Family']
        self.data_preprocess()
        # self.train_set = self.train_set.to_numpy()

    def data_preprocess(self):
        self.train_set.drop(['Family', 'Genus', 'Species', 'RecordID'], axis=1, inplace=True)
        self.train_set = MinMaxScaler().fit_transform(self.train_set)
        self.y = LabelEncoder().fit_transform(self.y)
        if self.select:
           self.train_set = SelectKBest(chi2, k=f_num).fit_transform(self.train_set, self.y)

    def print_result(self, result: dict, labels):
        print("Clustering result: ")
        print("cluster | numbers")
        for key, value in result.items():
            print("{:^7} | {:^8}".format(key, value))

        vs = v_measure_score(self.y, labels)
        ari = adjusted_rand_score(self.y, labels)
        ss = silhouette_score(self.train_set, labels, metric='euclidean')

        print("Result evaluating: ")
        print("Adjusted rand index: {:.6f}".format(ari))
        print("Silhouette coefficient: {:.6f}".format(ss))
        print("Average of homogeneity and completeness: {:.6f}".format(vs))

    def show_result(self, labels):
        print("Visualizing clustering result...")
        x_tsne = TSNE().fit_transform(self.train_set)
        x_min, x_max = x_tsne.min(0), x_tsne.max(0)
        # normalize
        x_norm = (x_tsne - x_min) / (x_max - x_min)

        plt.figure(figsize=(8, 8))
        for i in range(x_norm.shape[0]):
            plt.text(x_norm[i, 0], x_norm[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i]))
        plt.xticks([])
        plt.yticks([])
        plt.show()


# Agglomerative clustering
class MyAggClustering(BaseClustering):
    def __init__(self, train_set_path, k, s):
        BaseClustering.__init__(self, train_set_path, k, s)
        self.cls = AgglomerativeClustering(n_clusters=self.k, linkage='ward')

    def cluster(self):
        res = [0, 0, 0, 0]
        clusters = self.cls.fit(self.train_set)
        for i, cluster in enumerate(clusters.labels_):
            res[cluster] += 1
        res = {i: res[i] for i in range(len(res))}
        self.print_result(res, clusters.labels_)
        self.show_result(clusters.labels_)


class MyKMeansClustering(BaseClustering):
    def __init__(self, train_set_path, k, s):
        BaseClustering.__init__(self, train_set_path, k, s)
        self.centers = self.init_k_centers()

        self.iter = 10  # number of iterations
        self.best_labels = None
        self.best_result = None
        self.best_centers = None
        self.best_assessment = np.inf

    def init_k_centers(self):
        # pick the k initial cluster centers using k-means++
        centers = [random.choice(self.train_set)]
        for k in range(1, self.k):
            dist = np.array([min([np.inner(c - x, c - x) for c in centers]) for x in self.train_set])
            prob = dist / dist.sum()
            cumprob = prob.cumsum()
            cumprob[-1] = 1
            r = random.random()
            for i, p in enumerate(cumprob):
                if r < p:
                    centers.append(self.train_set[i])
                    break
        return centers

    def cluster(self):
        for i in range(self.iter):
            print("Iteration {} in progress...".format(i + 1))
            self._cluster()

        self.print_result(self.best_result, self.best_labels)
        self.show_result(self.best_labels)

    def _cluster(self):
        converged = False
        num = self.train_set.shape[0]         # number of entries
        labels = np.zeros(num, dtype=np.int)  # label of each entry
        assessment = np.zeros(num)            # used to assess the model

        while not converged:
            old_centers = np.copy(self.centers)
            for i in range(num):
                min_dist, min_idx = np.inf, -1
                for j in range(self.k):
                    dist = self._distance(self.train_set[i], self.centers[j])
                    if dist < min_dist:
                        min_dist, min_idx = dist, j
                        labels[i] = j
                assessment[i] = self._distance(self.train_set[i], self.centers[labels[i]])

            # update centers
            for i in range(self.k):
                self.centers[i] = np.mean(self.train_set[labels == i], axis=0)
            converged = self._converged(self.centers, old_centers)

        res = {}
        for key in np.unique(labels):
            res[key] = labels[labels == key].size

        assess = np.sum(assessment)
        if assess < self.best_assessment:
            self.best_result = res
            self.best_labels = labels
            self.best_assessment = assess
            self.best_centers = self.centers

    def _distance(self, sample1, sample2):
        return np.sum((sample1 - sample2) ** 2)

    def _converged(self, old_centers, centers):
        return set([tuple(c) for c in centers]) == \
            set([tuple(c) for c in old_centers])


def usage():
    print('Usage: {} -s -f/--file <train dataset file> -c/--clustering <clustering type> -'.format(sys.argv[0]))
    print('  -s, --select       select k-best features')
    print('  -f, --file         train dataset file path')
    print('  -c, --clustering   clustering type')
    print('      0 for MyAggClustering')


def parse_args(argv):
    global select
    global train_file
    global clustering
    try:
        opts, args = getopt.getopt(argv, 'hsf:c:', ['help', 'select','file=', 'clustering='])
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
        elif opt in ('-c', '--clustering'):
            if arg not in ('0', '1'):
                usage()
                sys.exit(1)
            clustering = int(arg)


def main():
    global select
    global train_file
    global clustering
    global n_clusters
    parse_args(sys.argv[1:])
    if clustering == 0:
        print("Using MyAggClustering to train the model...")
        MyAggClustering(train_file, n_clusters, select).cluster()
    elif clustering == 1:
        print("Using MyKMeansClustering to train the model..")
        MyKMeansClustering(train_file, n_clusters, select).cluster()


if __name__ == '__main__':
    main()

