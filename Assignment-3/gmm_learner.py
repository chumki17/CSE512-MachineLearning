import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn

class GMM:
    def __init__(self, C, n_runs):
        self.C = C  # number of Guassians/clusters
        self.n_runs = n_runs

    def get_params(self):
        return (self.mu, self.pi, self.sigma)

    def calculate_mean_covariance(self, X, prediction):
        d = X.shape[1]
        labels = np.unique(prediction)
        self.initial_means = np.zeros((self.C, d))
        self.initial_cov = np.zeros((self.C, d, d))
        self.initial_pi = np.zeros(self.C)

        counter = 0
        for label in labels:
            ids = np.where(prediction == label)  # returns indices
            self.initial_pi[counter] = len(ids[0])*1.0 / X.shape[0]*1.0
            X = np.asarray(X)
            self.initial_means[counter, :] = np.mean(X[ids], axis=0)
            de_meaned = X[ids] - self.initial_means[counter, :]
            Nk = X[ids].shape[0]  # number of data points in current gaussian
            self.initial_cov[counter, :, :] = np.dot(self.initial_pi[counter] * de_meaned.T, de_meaned) / Nk
            counter += 1
        return (self.initial_means, self.initial_cov, self.initial_pi)

    def _initialise_parameters(self, X):
        n_clusters = self.C
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=10, algorithm='auto')
        fitted = kmeans.fit(X)
        prediction = kmeans.predict(X)
        self._initial_means, self._initial_cov, self._initial_pi = self.calculate_mean_covariance(X, prediction)
        return (self._initial_means, self._initial_cov, self._initial_pi)

    def _e_step(self, X, pi, mu, sigma):
        N = X.shape[0]
        self.gamma = np.zeros((N, self.C))
        const_c = np.zeros(self.C)

        self.mu = self.mu if self._initial_means is None else self._initial_means
        self.pi = self.pi if self._initial_pi is None else self._initial_pi
        self.sigma = self.sigma if self._initial_cov is None else self._initial_cov

        for c in range(self.C):
            # Posterior Distribution using Bayes Rule
            self.gamma[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c], allow_singular = True)

        self.gamma= self.gamma*1.0
        # normalize across columns to make a valid probability
        gamma_norm = np.sum(self.gamma * 1.0, axis=1)[:, np.newaxis]
        self.gamma = self.gamma * 1.0 / gamma_norm * 1.0
        return self.gamma

    def _m_step(self, X, gamma):
        N = X.shape[0]
        C = self.gamma.shape[1]
        d = X.shape[1]

        self.pi = np.mean(self.gamma * 1.0, axis=0)*1.0

        self.mu = (np.dot(self.gamma.T, X) / np.sum(self.gamma, axis=0)[:, np.newaxis])*1.0

        for c in range(C):
            x = X - self.mu[c, :]

            gamma_diag = np.diag(self.gamma[:, c])
            x_mu = np.mat(x)
            gamma_diag = np.mat(gamma_diag)

            val = np.dot(x.T *1.0, gamma_diag*1.0)
            val2 = np.dot(val*1.0, x*1.0)
            sigma_c = val2
            self.gamma = self.gamma *1.0
            self.sigma[c, :, :] = sigma_c * 1.0 / (np.sum(self.gamma *1.0, axis=0)[:, np.newaxis][c])* 1.0

        return self.pi, self.mu, self.sigma

    def _compute_loss_function(self, X, pi, mu, sigma):
        N = X.shape[0]
        C = self.gamma.shape[1]
        self.loss = np.zeros((N, C))

        for c in range(C):
            self.mu[c] = self.mu[c]*1.0
            self.sigma[c] = self.sigma[c] * 1.0
            dist = mvn(self.mu[c], self.sigma[c], allow_singular=True)
            self.loss[:, c] = self.gamma[:, c] * (
                        np.log(self.pi[c] + 0.00001) + dist.logpdf(X) - np.log(self.gamma[:, c] + 0.000001))
        self.loss = np.sum(self.loss)
        return self.loss

    def fit(self, X):

        d = X.shape[1]
        self.mu, self.sigma, self.pi = self._initialise_parameters(X)

        try:
            for run in range(self.n_runs):
                self.gamma = self._e_step(X, self.mu, self.pi, self.sigma)
                self.pi, self.mu, self.sigma = self._m_step(X, self.gamma)
                loss = self._compute_loss_function(X, self.pi, self.mu, self.sigma)
                if run % 10 == 0:
                    print("Iteration: %d Loss: %0.6f" % (run, loss))

        except Exception as e:
            print("Exception is:", e)

        return self

    def predict(self, X):
        labels = np.zeros((X.shape[0], self.C))
        print("labels", labels)
        for c in range(self.C):
            labels[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c], allow_singular=True)
        labels = labels.argmax(1)
        return labels

    def predict_proba(self, X):
        post_proba = np.zeros((X.shape[0], self.C))
        for c in range(self.C):
            post_proba[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c], allow_singular=True)
            prob = np.sum(post_proba)
        return prob



def main():

    np.set_printoptions(suppress=True)
    parse = argparse.ArgumentParser(description='GMM commandline')

    # Add the arguments
    parse.add_argument('--components', type=str, help='no of components')
    parse.add_argument('--train', type=str, help='train data file path')
    parse.add_argument('--test', type=str, help='test data file path')

    args = parse.parse_args()
    components = args.components
    train_path = args.train
    test_path = args.test
    sum_proba =0


    train_data = pd.read_csv(train_path + "optdigits.train")
    test_data = pd.read_csv(test_path + "optdigits.test")

    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for digit in digits:
        train_data = train_data.loc[train_data['0.26'] == digit]
        if len(train_data) > 0:
            model = GMM(int(components), n_runs=10)
            fitted_values = model.fit(train_data)

    for digit in digits:
        test_data = test_data.loc[test_data['0.26'] == digit]
        if len(test_data) > 0:
            model = GMM(int(components), n_runs=10)
            fitted_values = model.fit(test_data)
            predict_proba  = model.predict_proba(test_data)
            print("Probability for digit", digit, "is:", predict_proba)
            sum_proba = sum_proba + predict_proba

    print ("Average probability" , sum_proba/10)


if __name__== '__main__':
    main()
