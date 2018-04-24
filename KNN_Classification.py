import numpy as np
from sklearn import preprocessing
import math
import operator
import collections
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import time

number_of_test_images = 174

class classication():

    f =71  #number of features

    def euclidean(self, X, Y):
        D = sum(pow((X[:self.f] - Y[:self.f]), 2))
        return math.sqrt(D)

    def neighborss (self, train_set, test_sample, k):
        dis = []
        neighbors = []
        for i in range(len(train_set)):
            dist = self.euclidean(test_sample, train_set[i, :self.f])
            dis.append((i, dist))
        dis.sort(key=operator.itemgetter(1))
        for j in range(k):
            t = dis[j][0]
            label = train_set[t, self.f]
            neighbors.append(label)
        return neighbors #best classes

    def assign_class(self, neighbor):
         X = collections.Counter(neighbor).most_common(1)
         return X[0][0]

    def _error(self, test, pred):
        correct = 0
        for i in range(len(test)):
            c = test[i][-1]
            if c == pred[i]:
                correct += 1
        return 100 - ((correct/float(len(test)))*100.0)

    def get_confusion(self, t, p):
        conv = confusion_matrix(t, p)
        df = pd.DataFrame(conv)
        print(df)
        accuracy = 0
        for i in range(len(df)):
            accuracy += conv[i][i]
        return accuracy


if __name__ == '__main__':
    start_time = time.time()
    p = classication()
    train = np.genfromtxt('Train_images_features.csv', delimiter=',')
    test = np.genfromtxt('Test_images_features.csv', delimiter=',')
    accuracy = []
    err_plot = []
    k_plot = []
    for k in range(1, 10):#choose the best number of neighbors
        pred = []
        for i in range(len(test)):
            n = p.neighborss(train, test[i], k)
            pred.append(p.assign_class(n))
        error = p._error(test, pred)
        accuracy.append((error, k))
        err_plot.append(error)
        k_plot.append(k)
        print(k)
    t = min(accuracy)
    U = test[:, [-1]]
    best_k = t[1]
   # accuracy = 100 - t[0]
    print("best K is= ", best_k)
    k = best_k
    best_pred = []
    plt.plot(k_plot, err_plot)
    plt.ylabel('error')
    plt.xlabel('K values')
    plt.show()
    for i in range(len(test)):
        n = p.neighborss(train, test[i], k)
        best_pred.append(p.assign_class(n))
    accuracy = (p.get_confusion(best_pred, U)/number_of_test_images)*100
    print("classification is Done successfully with total accuracy = "+str(accuracy)+'%')
    print("and total time = %s seconds " % (time.time() - start_time))
