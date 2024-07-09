#Importing the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

#Train the model

df_train=pd.read_csv('/content/Training.csv')
df_train=df_train.drop('Unnamed: 133',axis=1)

#Separate the dataset into independent and dependent values

X_train=df_train.iloc[:,:-1].values
y_train=df_train['prognosis'].values

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

#Apply the label encoder to transform the dependent variables

le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_val=le.fit_transform(y_val)

#calling the model using the object

from sklearn.svm import SVC
svc=SVC(kernel='poly',C=1.0,gamma='scale')
svc.fit(X_train,y_train)

y_pred=svc.predict(X_val)

#Visulaizing the results of the training model

plt.scatter(y_val,y_pred,color='red')
plt.title('Result of train model')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

accuracy=r2_score(y_val,y_pred)
print(accuracy)

#Testing the model

df_test=pd.read_csv('/content/Testing.csv')
X_test=df_test.iloc[:,:-1].values
y_test=df_test.iloc[:,-1].values
y_test=le.transform(y_test)

#Using R squared evaluting the performance of the model

y_pred=svc.predict(X_test)
accuracy=r2_score(y_test,y_pred)
print(accuracy)

#Build the confusion matrix for the model to evalute the prediction results

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('Actual')
plt.ylabel('predicted')
plt.show()

#Result of the test model

plt.scatter(y_test,y_pred)
plt.xticks(rotation=90,fontsize=10)
plt.yticks(fontsize=10)
plt.title('Result of test model')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

numb=np.random.randint(2,size=132)

a=svc.predict([numb])
le.inverse_transform(a)