#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    logisticRegressionClassifier = LogisticRegression(data.trainingSet,
                                                      data.validationSet,
                                                      data.testSet,
                                                      learningRate=0.0006,
                                                      epochs=200)

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nLogistic Regression has been training..")
    logisticRegressionClassifier.train()
    print("Done..")

    
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    #stupidPred = myStupidClassifier.evaluate()
    #perceptronPred = myPerceptronClassifier.evaluate()
    logRegPred = logisticRegressionClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the Logistic Regression:")
    # evaluator.printComparison(data.testSet, logRegPred)
    evaluator.printAccuracy(data.testSet, logRegPred)
    
    
if __name__ == '__main__':
    main()
