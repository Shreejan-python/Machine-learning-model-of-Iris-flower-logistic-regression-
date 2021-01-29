#predicting whether the flower is virginica or not by taking petal width in cm

#importing modules
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()
#print(iris)

#slicing petal width in cm
x = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)
#print(x, y)

#training
#if 0 comes it is not virginica and if 1 comes it is virginica
clf = LogisticRegression()
clf.fit(x, y)
for_example = clf.predict(([[2.3]]))
print(for_example)

#plotting the visualization
x_new = np.linspace(0, 3, 1000).reshape(-1,1)
#print(x_new)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new, y_prob[:, 1], 'g-', label='virginica')
plt.show()