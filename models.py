#%%
# import libraries
import pandas as pd
import numpy as np
import keras
import pandas_ml

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from pandas_ml import ConfusionMatrix
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.utils import np_utils

#%%
# read in data
data = pd.read_csv("final_data.csv")

#%%
# remove original instance information (URL, Domain, etc)
data = data.drop(columns=["URL", "WHOIS", "Protocol", "URL without Protocol", "Domain", "Number of Symbols", "Country"])

#%%
data.mean()

#%%
data.std()

#%%
# check number of malicious and benign examples
data_benign = data[data["Malicious"] == 0]
data_malicious = data[data["Malicious"] == 1]
print(data_benign.shape)
print(data_malicious.shape)

#%%
# create testing and training data
x_train, x_test, y_train, y_test = train_test_split(
    data.drop(columns="Malicious"), data["Malicious"], test_size=.3)
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

#%%
def train_test_split_with_index(x, y, train_index, test_index):
    """ Split both the x and y objects by the indices provided.
    Params:
        x: (DataFrame) Matrix representation of features for each instance
        y: (Series) Vector of result for each instance
        train_index: (ndarray) list of the indices that go to the training dataset
        test_index: (ndarray) list of the indices that go to the testing dataset
    Returns:
        x_test: (DataFrame) Features of instances in the testing set
        x_train: (DataFrame) Features of instances in the training set
        y_test: (DataFrame) Results of instances in the testing set
        y_train: (DataFrame) Results of instances in the training set
    """
    x_test = x.loc[test_index].reset_index(drop=True)
    x_train = x.loc[train_index].reset_index(drop=True)
    y_test = y.loc[test_index].reset_index(drop=True)
    y_train = y.loc[train_index].reset_index(drop=True)
    return x_test, x_train, y_test, y_train

#%%
def evaluate_model(expected, predicted):
    """ Calculates the various metrics to measure a model's performance and prints it
    Params (assumes values are listed in the same order for both params):
        expected: list of expected values
        predicted: list of predicted values
    """
    conf_matrix = confusion_matrix(expected, predicted)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # calculate metrics from confusion matrix
    true_positives = conf_matrix[1][1]
    false_positives = conf_matrix[0][1]
    true_negatives = conf_matrix[0][0]
    false_negatives = conf_matrix[1][0]
    accuracy = (true_positives + true_negatives)/(true_negatives + true_positives + false_negatives + false_positives)
    error = 1 - accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    
    # print metrics
    print("True positives: {}".format(true_positives))
    print("False positives: {}".format(false_positives))
    print("True negatives: {}".format(true_negatives))
    print("False negatives: {}".format(false_negatives))
    print("Accuracy: {}".format(accuracy))
    print("Error: {}".format(error))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 score: {}".format(f1_score))

#%% [markdown]
# # Logistic Regression

#%%
# test different probabilities for logistic regression model
kf = KFold(n_splits=10, shuffle=True)

for prob in [.5, .7, .8]:
    print("T is: {}".format(prob))
    accuracies = []
    for train_index, test_index in kf.split(x_train):
        x_validate, x_rest, y_validate, y_rest = train_test_split_with_index(x_train, y_train, train_index, test_index)
        
        log_reg = LogisticRegression(penalty="l1", solver="saga", n_jobs=-1)
        log_reg.fit(x_rest, y_rest)

        y_validate = pd.DataFrame(data=y_validate)
        y_rest = pd.DataFrame(data=y_rest)
        
        y_validate["Prob"] = pd.Series(log_reg.predict_proba(x_validate)[:,1])
        y_validate["Pred"] = np.where(y_validate["Prob"] >= prob, 1, 0)
        y_validate["Correct"] = np.where(y_validate["Pred"] == y_validate["Malicious"], 1, 0)
        accuracies.append(y_validate["Correct"].sum() / y_validate["Correct"].size)
    print("Average Accuracy: {}".format(pd.Series(data=accuracies).mean()))

#%%
# create logistic regression model fit with training data
log_reg = LogisticRegression(penalty="l1", solver="saga", n_jobs=-1)
log_reg.fit(x_train, y_train)

#%%
# prediction evaluation for testing data
print("T = 0.5")
evaluate_model(y_test, np.where(log_reg.predict_proba(x_test)[:,1] >= 0.5, 1, 0))
print()
print("T = 0.7")
evaluate_model(y_test, np.where(log_reg.predict_proba(x_test)[:,1] >= 0.7, 1, 0))
print()
print("T = 0.8")
evaluate_model(y_test, np.where(log_reg.predict_proba(x_test)[:,1] >= 0.8, 1, 0))

#%%
# check weights for logistic regression model
names = x_test.columns
coefs = log_reg.coef_[0]
sorted_coefs = dict()
for i in range(len(names)):
    sorted_coefs[names[i]] = coefs[i]

sorted_coefs = sorted(sorted_coefs.items(), key=lambda x: abs(x[1]), reverse=True)

for key, value in sorted_coefs:
    print("{}: {}".format(key, value))

#%% [markdown]
# # Random Forest

#%%
# test number of learners in Random Forest model
kf = KFold(n_splits=10, shuffle=True)

for learners in [50, 100, 200]:
    print("Learners: {}".format(learners))
    accuracies = []
    for train_index, test_index in kf.split(x_train):
        x_validate, x_rest, y_validate, y_rest = train_test_split_with_index(x_train, y_train, train_index, test_index)
        
        random_forest = RandomForestClassifier(n_estimators=learners, n_jobs=-1)
        random_forest.fit(x_rest, y_rest)
        
        y_validate = pd.DataFrame(data=y_validate)
        y_rest = pd.DataFrame(data=y_rest)
        
        y_validate["Pred"] = random_forest.predict(x_validate)
        y_validate["Correct"] = np.where(y_validate["Pred"] == y_validate["Malicious"], 1, 0)
        accuracies.append(y_validate["Correct"].sum() / y_validate["Correct"].size)
    print("Average Accuracy: {}".format(pd.Series(data=accuracies).mean()))

#%%
# create Random Forest with 50 learners and fit to training data
random_forest = RandomForestClassifier(n_estimators=50, n_jobs=-1)
random_forest.fit(x_train, y_train)

#%%
# prediction evaluation on the training data w/ 50 learners
evaluate_model(y_test, random_forest.predict(x_test))

#%%
# check feature importance for Random Forest
names = x_test.columns
importance = random_forest.feature_importances_
sorted_importance = dict()
for i in range(len(names)):
    sorted_importance[names[i]] = importance[i]

sorted_importance = sorted(sorted_importance.items(), key=lambda x: x[1], reverse=True)

for key, value in sorted_importance:
    print("{}: {}".format(key, value))

#%% [markdown]
# # Feed-Forward Neural Network

#%%
# create first feed-forward neural network (w/ 30% dropout)
ff_nn_1 = Sequential()

# hidden layer 1
ff_nn_1.add(layers.Dense(100, input_dim=132))
ff_nn_1.add(layers.Dropout(rate=0.3))
ff_nn_1.add(layers.Activation("relu"))

# hidden layer 2
ff_nn_1.add(layers.Dense(70))
ff_nn_1.add(layers.Dropout(rate=0.3))
ff_nn_1.add(layers.Activation("relu"))

# hidden layer 3
ff_nn_1.add(layers.Dense(45))
ff_nn_1.add(layers.Dropout(rate=0.3))
ff_nn_1.add(layers.Activation("relu"))

# hidden layer 4
ff_nn_1.add(layers.Dense(30))
ff_nn_1.add(layers.Dropout(rate=0.3))
ff_nn_1.add(layers.Activation("relu"))

# hidden layer 5
ff_nn_1.add(layers.Dense(20))
ff_nn_1.add(layers.Dropout(rate=0.3))
ff_nn_1.add(layers.Activation("relu"))

# hidden layer 5
ff_nn_1.add(layers.Dense(12))
ff_nn_1.add(layers.Dropout(rate=0.3))
ff_nn_1.add(layers.Activation("relu"))

# hidden layer 6
ff_nn_1.add(layers.Dense(8))
ff_nn_1.add(layers.Dropout(rate=0.3))
ff_nn_1.add(layers.Activation("relu"))

# output layer
ff_nn_1.add(layers.Dense(1))
ff_nn_1.add(layers.Activation("sigmoid"))

# compile network
ff_nn_1.compile(optimizer=RMSprop(), loss="binary_crossentropy", metrics=["accuracy"])

#%%
# train first feed-forward neural network
ff_nn_1.fit(x_train, y_train, epochs=100, batch_size=256)

#%%
# test first feed-forward neural network on testing data
print("T = 0.5")
evaluate_model(y_test, np.where(ff_nn_1.predict(x_test) >= 0.5, 1, 0))
print()
print("T = 0.7")
evaluate_model(y_test, np.where(ff_nn_1.predict(x_test) >= 0.7, 1, 0))
print()
print("T = 0.8")
evaluate_model(y_test, np.where(ff_nn_1.predict(x_test) >= 0.8, 1, 0))

#%%
# create second feed-forward neural network (w/ 30% dropout)
ff_nn_2 = Sequential()

# hidden layer 1
ff_nn_2.add(layers.Dense(100, input_dim=132))
ff_nn_2.add(layers.Dropout(rate=0.3))
ff_nn_2.add(layers.Activation("relu"))

# hidden layer 2
ff_nn_2.add(layers.Dense(50))
ff_nn_2.add(layers.Dropout(rate=0.3))
ff_nn_2.add(layers.Activation("relu"))

# hidden layer 3
ff_nn_2.add(layers.Dense(25))
ff_nn_2.add(layers.Dropout(rate=0.3))
ff_nn_2.add(layers.Activation("relu"))

# hidden layer 4
ff_nn_2.add(layers.Dense(12))
ff_nn_2.add(layers.Dropout(rate=0.3))
ff_nn_2.add(layers.Activation("relu"))

# hidden layer 5
ff_nn_2.add(layers.Dense(6))
ff_nn_2.add(layers.Dropout(rate=0.3))
ff_nn_2.add(layers.Activation("relu"))

# output layer
ff_nn_2.add(layers.Dense(1))
ff_nn_2.add(layers.Activation("sigmoid"))

# compile network
ff_nn_2.compile(optimizer=RMSprop(), loss="binary_crossentropy", metrics=["accuracy"])

#%%
# train second network
ff_nn_2.fit(x_train, y_train, epochs=100, batch_size=256)

#%%
# test second feed-forward neural network on testing data
print("T = 0.5")
evaluate_model(y_test, np.where(ff_nn_2.predict(x_test) >= 0.5, 1, 0))
print()
print("T = 0.7")
evaluate_model(y_test, np.where(ff_nn_2.predict(x_test) >= 0.7, 1, 0))
print()
print("T = 0.8")
evaluate_model(y_test, np.where(ff_nn_2.predict(x_test) >= 0.8, 1, 0))

#%%
