# Linear regression with multiple features using gradient decent
# The script will stop after itr number of iterations
# Written by: Intsar Saeed


import sys
import statistics
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class Classifier(object):

    def __init__(self, output_file=None, raw_data=None):
        self.raw_data = raw_data
        self.file = output_file

    def classify_data(self):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.raw_data[['A', 'B']], self.raw_data['label'])

        X_train, X_test, y_train, y_test = train_test_split(self.raw_data[['A', 'B']], self.raw_data['label'],
                                                            test_size=0.4, stratify=self.raw_data['label'])

        # Linear kernel SVM classifier
        linear_classifier_parameters = [{'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'kernel': ['linear']}]
        clf_grid = GridSearchCV(estimator=SVC(), param_grid=linear_classifier_parameters, n_jobs=-1)
        clf_grid.fit(X_train, y_train)
        tes_score = clf_grid.score(X_test, y_test)
        write_str = "svm_linear, " + str(clf_grid.best_score_) + ", " + str(tes_score) + "\n"
        self.file.write(write_str)
        print(write_str)

        # RBF kernel SVM classifier
        rbf__classifier_parameters = [{'C': [0.1, 0.5, 1, 5, 10, 50, 100],
                                       'gamma': [0.1, 0.5, 1, 3, 6, 10], 'kernel': ['rbf']}]
        clf_grid = GridSearchCV(estimator=SVC(), param_grid=rbf__classifier_parameters, n_jobs=-1)
        clf_grid.fit(X_train, y_train)
        tes_score = clf_grid.score(X_test, y_test)
        write_str = "svm_rbf, " + str(clf_grid.best_score_) + ", " + str(tes_score) + "\n"
        self.file.write(write_str)
        print(write_str)

        # k-Nearest Neighbors classifier
        knn_classifier_parameters = [{'n_neighbors': np.arange(1, 51, 1), 'leaf_size': np.arange(5, 61, 5)}]
        clf_grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_classifier_parameters, n_jobs=-1)
        clf_grid.fit(X_train, y_train)
        tes_score = clf_grid.score(X_test, y_test)
        write_str = "knn, " + str(clf_grid.best_score_) + ", " + str(tes_score) + "\n"
        self.file.write(write_str)
        print(write_str)

        # Logistic Regression classifier
        logistic_classifier_parameters = [{'C': [0.1, 0.5, 1, 5, 10, 50, 100]}]
        clf_grid = GridSearchCV(estimator=LogisticRegression(), param_grid=logistic_classifier_parameters, n_jobs=-1)
        clf_grid.fit(X_train, y_train)
        tes_score = clf_grid.score(X_test, y_test)
        write_str = "logistic, " + str(clf_grid.best_score_) + ", " + str(tes_score) + "\n"
        self.file.write(write_str)
        print(write_str)

        # Decision Trees classifier
        d_tree_classifier_parameters = [{'max_depth': np.arange(1, 51, 1), 'min_samples_split': np.arange(2, 11, 1)}]
        clf_grid = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=d_tree_classifier_parameters, n_jobs=-1)
        clf_grid.fit(X_train, y_train)
        tes_score = clf_grid.score(X_test, y_test)
        write_str = "decision_tree, " + str(clf_grid.best_score_) + ", " + str(tes_score) + "\n"
        self.file.write(write_str)
        print(write_str)

        # Polynomial kernel SVM classifier
        polynomial_classifier_parameters = [{'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'degree': [4, 5, 6],
                                             'gamma': [0.1, 0.5], 'kernel': ['poly']}]
        clf_grid = GridSearchCV(estimator=SVC(), param_grid=polynomial_classifier_parameters, n_jobs=-1)
        clf_grid.fit(X_train, y_train)
        tes_score = clf_grid.score(X_test, y_test)
        write_str = "svm_polynomial, " + str(clf_grid.best_score_) + ", " + str(tes_score) + "\n"
        self.file.write(write_str)
        print(write_str)

        # Random Forest classifier
        r_forest_classifier_parameters = [{'max_depth': np.arange(1, 51, 1), 'min_samples_split': np.arange(2, 11, 1)}]
        clf_grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=r_forest_classifier_parameters, n_jobs=-1)
        clf_grid.fit(X_train, y_train)
        tes_score = clf_grid.score(X_test, y_test)
        write_str = "random_forest, " + str(clf_grid.best_score_) + ", " + str(tes_score) + "\n"
        self.file.write(write_str)
        print(write_str)


# Main Function that reads in Input and Runs corresponding Algorithm
def main():

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    f = open(output_file, 'w')

    # Read the data
    # age in years, weight in KG, height in meters
    data = pd.read_csv(input_file, header=0)

    # Load the data
    classifier = Classifier(output_file=f, raw_data=data)
    classifier.classify_data()

    # Close the file
    print("Completed")
    f.close()


if __name__ == '__main__':
    main()
