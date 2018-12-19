import numpy as np
from sklearn import tree
from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Basic scikit decision tree with custom input
# [height, weight, shoe size] [inches, pounds, US]
X = [[67, 150, 10], [63, 130, 6], [66, 140, 7], [72, 170, 11], [60, 115, 6], [69, 155, 10], [74, 180, 12], [70, 175, 10], [68, 150, 8], [65, 140, 7], [76, 200, 13], [72, 165, 10], [64, 150, 8], [63, 120, 5], [62, 160, 6], [67, 210, 9]]
Y = ['male', 'female', 'female', 'male', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'female', 'male']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, Y)

prediction = clf.predict([[69, 160, 8], [72, 180, 10], [67, 160, 9], [67, 150, 7]])

print (prediction)

# Different scikit models with larger dataset
# Load data
digits = datasets.load_digits()
imagesAndLabels = list(zip(digits.images, digits.target))
nSamples = len(digits.images)
data = digits.images.reshape((nSamples, -1))

# Identify classifiers
classifierNames = ['KNN', 'Linear SVM', 'Decision Tree', 'Random Forest', 'Neural Net', 'Naive Bayes']

classifiers = [KNeighborsClassifier(3), SVC(kernel='linear', C=0.025), DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               MLPClassifier(alpha=1, max_iter=500, activation='logistic', solver='adam'), GaussianNB()]

# Split data
xTrain = data[:nSamples // 2]
yTrain = digits.target[:nSamples // 2]
xTest = data[nSamples // 2:]
yTest = digits.target[nSamples // 2:]

# Iterate through each of the classifiers and train on digit dataset
for name, classifier in zip(classifierNames, classifiers):
    classifier.fit(xTrain, yTrain)
    expected = yTest
    predicted = classifier.predict(xTest)
    print('Classification report for classifier: ' + name + '\n' + metrics.classification_report(expected, predicted))




