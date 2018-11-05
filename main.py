import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("datasets/train.csv").as_matrix()

#classifier 
clf = DecisionTreeClassifier()

#traning classifier and train data
x_train = data[0:21000,1:]
x_lables = data[0:21000,0]

clf.fit(x_train,x_lables)

#test data 
x_test = data[21000:,1:]
x_lables = data[21000:,0]

#Given test input as '3' (variable = d)
d = x_test[56]
d.shape=(28,28)
#print(d)
#will show the selected input vai plot
pt.imshow(255-d,cmap='gray')

#prediction given by classifier 
print("Predicted Output")
print(clf.predict( [ x_test[56] ] ))

p = clf.predict(x_test)
count = 0
for i in range(0,21000):
    count+=1 if p[i] == x_lables[i] else 0
print("Accuracy count",(count/21000)*100)

pt.show()

