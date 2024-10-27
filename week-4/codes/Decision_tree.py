import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("drug200.csv")
data.head(5)

"""The sodium/potassium adenosine-triphosphatase (Na+/K+-ATPase) is an essential plasma membrane enzyme that maintains ion homeostasis, cell volume and contractility, electrical signaling, membrane trafficking and vascular tone (7). The Na+/K+-ATPase is the target of several controlling mechanisms."""

le = LabelEncoder()

data['gender'] = le.fit_transform(data['gender'])

data['BP'] = le.fit_transform(data['BP'])

data['Cholesterol'] = le.fit_transform(data['Cholesterol'])

data.head(5)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size= 0.8, random_state = 0)

drugTree = DecisionTreeClassifier(criterion="gini",max_depth = 4)

model = drugTree.fit(X_train, y_train)

y_pred = drugTree.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

from sklearn import tree

tree.plot_tree(model)
plt.show()

fig = plt.figure(figsize=(15,15))
_ = tree.plot_tree(model,
                   feature_names= data.columns,
                   class_names= data.Drug.unique(),
                   filled=True)



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 2, criterion = 'gini', random_state = 0)

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
