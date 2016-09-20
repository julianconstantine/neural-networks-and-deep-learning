"""
mnist_svm
---------

"""

# Libraries
import mnist_loader

from sklearn import svm


def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()

    # Initialize a support vector classifier
    clf = svm.SVC()

    # Train the classifier
    clf.fit(X=training_data[0], y=training_data[1])

    # Get predictions
    predictions = [int(a) for a in clf.predict(X=test_data[0])]

    num_correct = sum(a == y for a, y in zip(predictions, test_data[1]))

    print("Baseline classifier using an SVM")
    print("%s of %s values correct" % (num_correct, len(test_data[1])))

