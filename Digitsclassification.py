from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
digits=load_digits()

X=digits.data
y=digits.target

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)

model=LogisticRegression()
model.fit(X_train,y_train)

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[8])

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[8])

#Testing the model with Handwritten numbers

from PIL import Image, ImageOps
import numpy as np

img=Image.open('C:/Users/HP/Desktop/sampleimg1.png').convert('L')
img_inverted= ImageOps.invert(img)

arr=np.array(img_inverted)
print(arr)

arr1=arr.flatten()

model.predict([arr1])

y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred,y_test)
print(cm)