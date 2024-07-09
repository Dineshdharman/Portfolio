import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('/content/breast-cancer.csv')
df.columns
df.isnull().sum()
df.drop(['id'],axis=1,inplace=True)
df

X=df.iloc[:,1:-1]
y=df.iloc[:,0]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


score=[]
for k in range(1,14):
  model=KNeighborsClassifier(n_neighbors=k,metric='euclidean')
  model.fit(X_train,y_train)
  y_pred=model.predict(X_test)
  score.append(accuracy_score(y_test,y_pred))
  print('k=%d,score=%2.f%%'%(k,score[k-1]*100))

import matplotlib.pyplot as plt
plt.plot(range(1,14),score)
plt.xlabel('value of k')
plt.ylabel('Accuracy')

model=KNeighborsClassifier(n_neighbors=11,metric='euclidean')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)

test_data=[
    [20.57, 17.77, 132.9, 1326.0, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667, 0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532, 24.99, 23.41, 158.8, 1956.0, 0.1238, 0.1866, 0.2416, 0.186, 0.275],
    [12.45, 15.7, 82.57, 477.1, 0.1278, 0.17, 0.1578, 0.08089, 0.2087, 0.07613, 0.3345, 0.8902, 2.217, 27.19, 0.00751, 0.03345, 0.03672, 0.01137, 0.02165, 0.005082, 15.47, 23.75, 103.4, 741.6, 0.1791, 0.5249, 0.5355, 0.1741, 0.3985],
    [13.71, 20.83, 90.2, 577.9, 0.1189, 0.1645, 0.09366, 0.05985, 0.2196, 0.07451, 0.5835, 1.377, 3.856, 50.96, 0.008805, 0.03029, 0.02488, 0.01448, 0.01486, 0.005412, 17.06, 28.14, 110.6, 897.0, 0.1654, 0.3682, 0.2678, 0.1556, 0.3196],
    [15.3, 25.27, 102.4, 732.4, 0.1082, 0.1697, 0.1683, 0.08751, 0.1926, 0.0654, 0.439, 1.012, 3.498, 43.5, 0.005233, 0.03057, 0.03576, 0.01083, 0.01768, 0.002967, 20.27, 36.71, 149.3, 1269.0, 0.1641, 0.611, 0.6335, 0.2024, 0.4027],
    [13.03, 18.42, 82.61, 523.8, 0.08983, 0.03766, 0.02562, 0.02923, 0.1467, 0.05863, 0.1839, 2.342, 1.17, 14.16, 0.004352, 0.004899, 0.01343, 0.01164, 0.02671, 0.001777, 13.3, 22.81, 84.46, 545.9, 0.09701, 0.04619, 0.04833, 0.05013, 0.1987]
]

for i in test_data:
  print(model.predict([i]))