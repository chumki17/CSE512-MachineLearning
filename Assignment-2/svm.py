import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import argparse
import gzip
from mnist import MNIST
import os

def calculate_cost_gradient(W, X, Y):

    Y = np.array([Y])
    X = np.array([X])

    distance = 1 - (Y * np.dot(X, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if (max(0, d) == 0):
            di = W
        else:
            di = W - (reg_strength * Y[ind] * X[ind])
        dw += di
    dw = dw/len(Y)
    return dw


def sgd(features, outputs, weights):
    max_epochs = 10

    for epoch in range(1, max_epochs):

        X, Y = features, outputs
        for (ind, x) in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - learning_rate * ascent

    return weights

def compute_sgd_accuracy(X,Y ,W):
    y_predicted = np.array([])
    count = 0
    for i in range(len(X)):
        yp = np.sign(np.dot(W, X[i]))  # model
        if yp == Y[i]:
            count += 1
        y_predicted = np.append(y_predicted, yp)

    error = (len(Y) - count) * 1.0 / len(Y) * 1.0
    accuracy = accuracy_score(Y, y_predicted)
    return error, accuracy


def comput_K (xi, xj):
    gamma = 0.7
    val = np.exp(-gamma * np.linalg.norm(xi - xj) ** 2)
    return val

def rbf(X,Y, X_count):

    X = np.array(X)
    Y = np.array(Y)

    alpha = np.zeros(X_count)

    # Calculating Gram matrix
    K = np.zeros((X_count, X_count))
    for i in range(X_count):
        for j in range(X_count):
            K[i, j] = comput_K(X[i], X[j])

    max_iterations = 100
    for ite in range(max_iterations):
        for i in range(X_count):
            sum = 0
            val = 0
            for j in range(X_count):
                val= alpha[j] * Y[j] * K[i,j]
                sum = sum + val
            if sum <= 0:
                val = -1
            elif sum >0:
                val = 1
            if val != Y[i]:
                alpha[i] = alpha[i] + 1
    return alpha

def compute_rbf_accuracy(train_X,train_Y,test_X,test_Y,alpha):
    m = test_Y.size
    count = 0
    for i in range(m):
        y_predict = 0
        for a, x_train,y_train  in zip(alpha, train_X,train_Y):
            y_predict += a * y_train * comput_K(test_X[i],x_train)
        if y_predict > 0:
            y_predict = 1
        elif y_predict <= 0:
            y_predict = -1
        if test_Y[i] == y_predict:
            count +=1
    error = (len(test_Y) - count) * 1.0 / len(test_Y) * 1.0
    accuracy = 1 - error
    return error, accuracy

def main():
    np.set_printoptions(suppress=True)
    parse = argparse.ArgumentParser(description='SVM commandline')

    # Add the arguments
    parse.add_argument('--kernel', type=str, help='linear or RBF')
    parse.add_argument('--dataset', type=str, help='dataset filenam (bsd or mnist)')
    parse.add_argument('--input', type=str, help='input file path')
    parse.add_argument('--output', type=str, help='output file path')
    args = parse.parse_args()
    kernel = args.kernel
    dataset = args.dataset
    input_file_path = args.input
    op_file_path = args.output



    if (dataset == "bcd"):

        data = pd.read_csv(input_file_path + "Breast_cancer_data.csv")
        Y = data.loc[:, 'diagnosis']
        data = data.drop(['diagnosis'], axis=1)
        X = data.iloc[:, 1:]
        X_normalized = MinMaxScaler().fit_transform(X.values)
        X = pd.DataFrame(X_normalized)

        if(kernel == "linear"):

            X.insert(loc=len(X.columns), column='intercept', value=1)

            weights = np.zeros(X.to_numpy().shape[1])
            W = sgd(X.to_numpy(), Y.to_numpy(), weights)
            print ('Weights are: {}'.format(W))
            err, acc = compute_sgd_accuracy(X.to_numpy(), Y.to_numpy(), W)
            print ("Training error :", err, "Accuracy :", acc)

            f = open(op_file_path+"output.txt", "w")
            f.write('Weights are: {}'.format(W))
            f.close()

        elif (kernel == "rbf"):

            X_count = X.shape[0]
            W = rbf(X.to_numpy(), Y.to_numpy(), X_count)
            err, acc = compute_rbf_accuracy(X.to_numpy(), Y.to_numpy(), X.to_numpy(), Y.to_numpy(), W)
            print ("Training error : ", err, "Accuracy :", acc)

            f = open(op_file_path+"output.txt", "w")
            f.write('Weights are: {}'.format(W))
            f.close()

    elif (dataset == "mnist"):

        mndata = MNIST(input_file_path + "samples")
        images, labels = mndata.load_training()
        img, lbl = mndata.load_testing()
        img_size = 28
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_sum = 0
        test_sum = 0

        if (kernel == "linear"):

            for digit in digits:
                train_labels =[]
                test_labels =[]
                for i in range(0, len(labels)):
                    if(labels[i] == digit):
                        train_labels.append(1)
                    else:
                        train_labels.append(0)

                for i in range(0, len(lbl)):
                    if(lbl[i] == digit):
                        test_labels.append(1)
                    else:
                        test_labels.append(0)

                weights = np.zeros(img_size* img_size)
                W = sgd(images, train_labels, weights)
                err, acc = compute_sgd_accuracy(images, train_labels,W)
                print ("Training error : ", err, "Accuracy :", acc)
                train_sum += err
                test_err, test_acc = compute_sgd_accuracy(img, test_labels, W)
                print ("Testing error : ", test_err, "Accuracy :", test_acc)
                test_sum += test_err

                f = open(op_file_path+"output.txt", "w")
                f.write('Weights are: {}'.format(W))
                f.close()

            avg_train_err = train_sum*1.0 / len(digits)
            avg_test_err = test_sum * 1.0 / len(digits)
            print ("Average training error : ", avg_train_err, "Average testing error :", avg_test_err)

        elif (kernel == "rbf"):
            X_count = len(images)
            for digit in digits:

                train_labels =[]
                test_labels =[]

                for i in range(0, len(labels)):
                    if(labels[i] == digit):
                        train_labels.append(1)
                    else:
                        train_labels.append(0)

                for i in range(0, len(lbl)):
                    if(lbl[i] == digit):
                        test_labels.append(1)
                    else:
                        test_labels.append(0)

                W = rbf(images , train_labels , X_count)
                err, acc = compute_rbf_accuracy(images, train_labels, img, test_labels, W)
                print ("Training error : ", err, "Accuracy :", acc)

                f = open(op_file_path+"output.txt", "w")
                f.write('Weights are: {}'.format(W))
                f.close()

if __name__== '__main__':
    reg_strength = 10000  # regularization strength
    learning_rate = 1
    main()
