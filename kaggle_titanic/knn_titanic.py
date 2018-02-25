"""
"""

print("*** KNN Iris Starting ***\n")

# Library
import os
import random
import pandas as pd
from sklearn import neighbors
from sklearn import metrics
from sklearn import utils

# Setting work dir
scriptloc = os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptloc)

# Import Data
traindf = pd.read_csv('train.csv')
testdf = pd.read_csv('test.csv')

# Dealing with NaN values
## Based on traindf.Age.describe(), sum(traindf.Age.isnull()) is not 0
## there is some null values.
## Impute with 0
traindf.Age = traindf.Age.fillna(0)
print(traindf.Age)

# Data Exploration
## Sex vs Survival
sex_survived = pd.crosstab(traindf.Survived, traindf.Sex)

##  PClass vs Survival
pclass_survived = pd.crosstab(traindf.Survived, traindf.Pclass)

## Pclass & Sex vs Survival
sexclass_survived = pd.crosstab(traindf.Survived, [traindf['Sex'], traindf['Pclass']])



# Data Processing

# Feature Engineering
## if sex is male then 0, if female then 1
## Use 'map' for string replacement
traindf['Sex'] = traindf['Sex'].map({'female':1, 'male':0})
testdf['Sex'] = testdf['Sex'].map({'female':1, 'male':0})


# Splitting data to train and test
split = 0.7
traindf = utils.shuffle(traindf)
idx = round(split * traindf.shape[0])
traindf_train = traindf.iloc[:idx]
traindf_test = traindf.iloc[idx:]


# Feature Selection
# Feature
features = ['Sex', 'Pclass', 'Age']

# Data Structure conversion
## Converting columns in feature dataframe to list of list
traindf_trainX = traindf_train[features].values.tolist()
traindf_testX = traindf_test[features].values.tolist()
#testdf_X = testdf[features].values.tolist()

# Target Var
traindf_trainY = traindf_train.Survived.tolist()
traindf_testY = traindf_test.Survived.tolist()
#testdf_y = testdf.Survived


# Modelling
print(traindf_trainX)
# Predict based on whether the passenger is female
knn = neighbors.KNeighborsClassifier(n_neighbors=11)
knn.fit(traindf_trainX, traindf_trainY)
prediction = knn.predict(traindf_testX)

# Accuracy
accuracy = metrics.accuracy_score(traindf_testY, prediction)
print("Accuracy: " + repr(accuracy*100.0))


# Printing
print("*** PRINTOUT ***\n")
#print(sex_survived)
#print(sexclass_survived)


# Output result
print("*** OUTPUT RESULT ***\n")
# myoutput = pd.DataFrame({"PassengerId": testdf.PassengerId,
#                          "Survived":prediction})


## Write result to csv
#myoutput.to_csv('my_submission2.csv', index=False)

## Detailed result df
# features.append('PassengerId')
# detaildf = testdf[features]
# detaildf['Survived'] = prediction
# #detaildf.assign(Survived = lambda x: prediction)
# detaildf.to_csv('detailsubmission.csv', index=False)


## End
print("\n*** KNN Iris Ending ***")