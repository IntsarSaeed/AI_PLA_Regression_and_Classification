# Linear regression with multiple features using gradient decent
# The script will stop once the solution converges
# Written by: Intsar Saeed


import sys
import statistics
import pandas as pd
import numpy as np


class LinearRegression(object):

    """
    The class for linear regression (using gradient decent)
    :param itr: Number of iterations over the training set
    :param raw_data: The raw data on which the linear regression is to be performed
    """
    def __init__(self, output_file=None, raw_data=None, itr=100):
        self.raw_data = raw_data
        self.itr = itr
        self.file = output_file

        self.alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 5, 10]  # learning Rate
        self.betas = np.zeros(len(list(self.raw_data)))  # betas (or weights)
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.cost = np.zeros(2)

    def normalize_data(self):
        """
        This method normalizes the data in all the columns and save the normalized data
        The matrices X and Y will be updated with the normalised data
        """
        for i in range(0, len(list(self.raw_data))):
            mean = statistics.mean(self.raw_data[i])
            std = statistics.stdev(self.raw_data[i])
            self.X[0] = np.ones(len(self.raw_data))  # add a column of ones

            if i != (len(list(self.raw_data))-1):
                self.X[i+1] = (self.raw_data[i] - mean) / std
            else:
                #self.Y = (self.raw_data[i] - mean) / std
                self.Y = self.raw_data[i]
        return

    def train(self):
        """
        Method to train the date
        :returns: The new values of the betas
        """
        n = len(self.raw_data)  # np of data inputs
        for i in range(0, len(self.alpha)):

            if self.alpha[i] != 1.5:
                num_of_itr = self.itr
            else:
                num_of_itr = 15

            for _ in range(num_of_itr):
                f_x = self.X.dot(self.betas)  # predicted (hypothesis)
                cost = -1 * (self.Y - f_x)  # Actual - hypothesis
                #print(cost)
                #print(self.X)
                grd = self.X.T.dot(cost) / n
                self.betas = self.betas - (self.alpha[i] * grd)
            print("alpha = ", float(self.alpha[i]), ", itr = ", float(num_of_itr), ", b0 = ", float(self.betas[0]),
                  ", b1 = ", float(self.betas[1]), ", b2 = ", float(self.betas[2]))
            write_str = "%s, %s, %s, %s, %s\n" % (self.alpha[i], num_of_itr, self.betas[0], self.betas[1], self.betas[2])
            self.file.write(write_str)
        return


# Main Function that reads in Input and Runs corresponding Algorithm
def main():

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    f = open(output_file, 'w')

    # Read the data
    # age in years, weight in KG, height in meters
    data = pd.read_csv(input_file, header=None)

    # Load the data
    linear_regression = LinearRegression(output_file=f, raw_data=data)
    linear_regression.normalize_data()
    linear_regression.train()


if __name__ == '__main__':
    main()
