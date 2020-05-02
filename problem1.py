# This script implements a perceptron learning algorithm using a step function
# The script will stop once the solution converges
# Written by: Intsar Saeed

import sys
import numpy as np
import pandas as pd


class Perceptron(object):

    """
    The class representing a basic perceptron
    :param alpha: learning rate
    :param itr: Number of iterations over the training set
    """
    def __init__(self, alpha=0.1, itr=50, output_file=None):
        self.alpha = alpha  # learning rate
        self.itr = itr  # Number of iteration
        self.weights = []
        self.file = output_file
        self.converged = False
        self.converge_count = 0

    def train(self, inputs, labels):
        """
        Method to train the date
        :parameter inputs: the inputs to the perceptron
        :parameter labels: the defined (known) label for a corresponding input
        :returns: The calculated new weights of the Network
        """

        self.weights = np.zeros(inputs.shape[1] + 1)
        last_weights = self.weights.copy()
        # self.weights = [6.80000, -0.8, -0.399999]

        for _ in range(self.itr):
            for i in range(0, len(inputs)):

                output = self.step_input([inputs.get('X1')[i], inputs.get('X2')[i]])
                # updated weight += alpha * (Yk - Ok) * Xk
                self.weights[1] += self.alpha * (labels[i] - output) * inputs.get('X1')[i]
                self.weights[2] += self.alpha * (labels[i] - output) * inputs.get('X2')[i]
                self.weights[0] += self.alpha * (labels[i] - output)

            print("w1 = ", float(self.weights[1]), ", w2 = ", float(self.weights[2]), ", b = ", float(self.weights[0]))
            write_str = "%s, %s, %s\n" % (self.weights[1], self.weights[2], self.weights[0])
            self.file.write(write_str)

            if list(self.weights) == list(last_weights):
                self.converge_count += 1
                if self.converge_count >= 2:
                    print("Solution Converged!")
                    self.converged = True
                    break
            else:
                last_weights = self.weights.copy()
                self.converge_count = 0
        return

    def step_input(self, value):
        """
       Method to taring the date
       :parameter value: the inputs to the step function
       :returns: The activation bool of the neural network
       """
        f_xi = np.dot(value, self.weights[1:]) + self.weights[0]  # f(x_i) summation
        if f_xi > 0:
            activation = 1
        else:
            activation = -1
        return activation


# Main Function that reads in Input and Runs corresponding Algorithm
def main():

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    f = open(output_file, 'w')

    # Read the data
    data = pd.read_csv(input_file, names=['X1', 'X2', 'Y'], header=None)

    # Train the data
    perceptron = Perceptron(output_file=f)
    perceptron.train(data[['X1', 'X2']], data.get('Y'))

    # Close the file
    print("Completed")
    f.close()


if __name__ == '__main__':
    main()
