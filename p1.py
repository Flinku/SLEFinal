import numpy as np
import pandas
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
import keras

rawDat = pandas.read_csv("College.csv", sep=",")
# Removes the name column, as that's useless to us
rawDat.pop("Name")

# Removes the Is Private column, since that's used for our y-values
rawY = rawDat.pop("Private")
# Replaces yes/no inputs with 1/0, with 1 representing private schools
rawY = rawY.replace({'Yes': 1, 'No': 0})

# Converts from pandas dataframe to numpy array
x_np = rawDat.to_numpy()
y_np = rawY.to_numpy()

# Splits the data into training and test
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_np, y_np, test_size=0.1)

# Scales the data to a range from 0-1. Used by ANN
MMscaler = preprocessing.MinMaxScaler()
MMscaler.fit(x_np)
x_MM = MMscaler.transform(x_train)
x_MM_test = MMscaler.transform(x_test)

# Scales the data about the standard. Used by linreg and logit
Sscaler = preprocessing.StandardScaler()
Sscaler.fit(x_np)
x_S = Sscaler.transform(x_train)
x_S_test = Sscaler.transform(x_test)


# -------------LINEAR REGRESSION---------------------
# Alphas tested
alphas = [0.1, 0.5, 1, 5, 10, 25, 50, 100]
# Performs a ridge regression with all the alphas. Uses the built-in ridge cross-validation function
ridgeReg = linear_model.RidgeClassifierCV(alphas=alphas, cv=5)
ridgeReg.fit(x_S, y_train)
bestAlpha = ridgeReg.alpha_


# -------------LOGISTIC REGRESSION-------------------
# C's tested
Cs = [0.1, 1, 10, 100, 1000, 10000]
# Different solvers to be tested. Newton-Cholesky is excluded because it doesn't handle the C optimization well, and
# it's just a poor choice for the data set
solvers = ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
# Initializes the max values so we can save them
maxScore = 0
maxC = 0.1
maxSolver = ""
# Iterates over all of the solvers, which then iterate over all the C values to find the best logit model
for solver in solvers:
    logReg = linear_model.LogisticRegressionCV(Cs=Cs, solver=solver, max_iter=10000, cv=5)
    logReg.fit(x_S, y_train)
    # If the current model is a better fit than the previous best, saves the hyperparameters
    thisScore = logReg.score(x_S, y_train)
    if thisScore > maxScore:
        maxScore = thisScore
        maxC = logReg.C_[0]
        maxSolver = solver


# -----------ARTIFICIAL NEURAL NETWORK----------------
# Units on the first and second layers in the different neural networks used
layer1Units = [0, 10, 20]
layer2Units = [0, 10, 20]

# Generates an ANN with up to 2 hidden layers with a given training set
def genANN(layer1, layer2, trainingDat, trainingY):
    model = keras.models.Sequential()
    if layer1 > 0:
        model.add(keras.layers.Dense(units=layer1, activation='tanh'))
    if layer2 > 0:
        model.add(keras.layers.Dense(units=layer2, activation='tanh'))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')
    model.fit(trainingDat, trainingY, epochs=200, verbose=0)
    return model


# Splits the training data for cross-validation
numSplits = 5
skf = model_selection.StratifiedKFold(n_splits=numSplits)

# Initializes best values
bestL1 = 0
bestL2 = 0
bestAcc = 0

for i in layer1Units:
    for j in layer2Units:
        # Keeps track of the cumulative accuracy score of each fold
        accSum = 0
        # Runs the model on each fold, and adds the accuracy of that specific model to the sum
        for k, (train_index, test_index) in enumerate(skf.split(x_MM, y_train)):
            newModel = genANN(layer1=i, layer2=j, trainingDat=x_MM[train_index], trainingY=y_train[train_index])
            newLoss, newAcc = newModel.evaluate(x_MM[test_index], y_train[test_index])
            accSum += newAcc
        # If cumulative accuracy of this model is greater than the previous best, marks these hyperparams as the best
        if accSum > bestAcc:
            bestL1 = i
            bestL2 = j
            bestAcc = accSum


# With best hyperparameters set, we can now build the final models and compare to the test data
# Uses found hyperparams, constructs a new model, and uses the accuracy function to see which model is the most precise

# LINEAR REGRESSION
ridgeRegFin = linear_model.RidgeClassifier(alpha=bestAlpha)
ridgeRegFin.fit(x_S, y_train)
ridgeScore = ridgeRegFin.score(x_S_test, y_test)

# LOGIT
logRegFin = linear_model.LogisticRegression(C=maxC, solver=maxSolver)
logRegFin.fit(x_S, y_train)
logitScore = logRegFin.score(x_S_test, y_test)

# ANN
annFin = genANN(layer1=bestL1, layer2=bestL2, trainingDat=x_MM, trainingY=y_train)
finLoss, finAcc = annFin.evaluate(x_MM_test, y_test)

# Prints summaries of models
print(f"LinReg: alpha={bestAlpha}\tAccuracy={ridgeScore}")
print(f"Logit: C={maxC}, solver={maxSolver}\tAccuracy={logitScore}")
print(f"ANN: First Layer={bestL1}, Second Layer={bestL2}\tAccuracy={finAcc}")

# Prints out which model is the best
if ridgeScore > logitScore and ridgeScore > finAcc:
    print("Linear Regression is the best model here")
elif logitScore > ridgeScore and logitScore > finAcc:
    print("Logistic Regression is the best model here")
elif finAcc > ridgeScore and finAcc > logitScore:
    print("Artificial Neural Network is the best model here")
else:
    print("There was a tie")

