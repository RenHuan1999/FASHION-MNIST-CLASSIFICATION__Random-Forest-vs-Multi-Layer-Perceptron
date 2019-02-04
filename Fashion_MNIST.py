#!/usr/bin/env python
# coding: utf-8

# In[72]:


import tensorflow as tf
from tensorflow import keras
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[4]:


train_images.shape, train_labels.shape, test_images.shape, test_labels.shape


# In[20]:


some_item = 2345
print(class_names[train_labels[some_item]])
plt.figure()
plt.imshow(train_images[some_item])


# In[21]:


train_images = train_images / 255.0

test_images = test_images / 255.0


# In[23]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


# In[63]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])


# In[64]:


model.fit(train_images, train_labels, epochs=50)


# In[65]:


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


# In[100]:


dnn_predictions = model.predict(test_images)


# In[88]:


train_images1 = train_images.flatten().reshape(60000,784)

forest_clf = ensemble.RandomForestClassifier(n_estimators=100)

forest_clf.fit(train_images1,train_labels)


# In[89]:


test_images1 = test_images.flatten().reshape(10000,784)
print(forest_clf.score(test_images1, test_labels))
forest_predictions = forest_clf.predict(test_images1)


# In[108]:


some_item = 230

plt.figure()
plt.imshow(test_images[some_item])

print("Perceptron Prediction")
print(class_names[np.argmax(dnn_predictions[some_item])])
print("\nRandom Forest Prediction")
print(class_names[forest_predictions[some_item]])
#print(np.around(predictions[some_item],3))
print("\nActual Class")
print(class_names[test_labels[some_item]])


# In[ ]:




