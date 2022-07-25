#!/usr/bin/env python
# coding: utf-8

# # Data

# In[1]:


import tensorflow as tf

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

mnist = tf.keras.datasets.mnist  #images hand-written digits from 0 to 9 "28x28"

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy', 
             metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)


# # Visualize Examples

# In[2]:


num_classes = 10
f, ax = plt.subplots(1, num_classes, figsize=(20,20))

for i in range(0, num_classes):
    sample = x_train[y_train == i][0]
    ax[i].imshow(sample, cmap='gray')
    ax[i].set_title("Label: {}".format(i), fontsize=16)


# In[3]:


for i in range(10):
    print(y_train[i])


# # Evaluate Loss and Accuracy

# In[4]:


test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss, test_acc)


# In[5]:


import matplotlib.pyplot as plt

plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

print (x_train[0])


# # Visualize Example Maping

# In[6]:


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[7]:


for i in range(10):
    print(y_train[i])


# In[8]:


model.save('MNIST_DATASET_PREDICTION')


# In[9]:


new_model = tf.keras.models.load_model('MNIST_DATASET_PREDICTION')


# # Train and Testing

# In[10]:


y_predictions = new_model.predict([x_test])
y_predictions_classes = np.argmax(y_predictions, axis = 1)


# In[11]:


print(y_predictions)
print(y_predictions_classes)


# In[12]:


import numpy as np

print(np.argmax(y_predictions[0]))


# In[13]:


plt.imshow(x_test[0])
plt.show()


# # Evaluate Prediction

# In[26]:


random_idx = np.random.choice(len(x_test))
x_sample = x_test[random_idx]
y_true = np.argmax(y_test, axis=1)
y_sample_true = y_true[random_idx]
y_sample_predictions_class = y_predictions_classes[random_idx]

plt.title("Predicted: {}, True: {}".format(y_sample_predictions_class, y_sample_true), fontsize=16)
plt.imshow(x_sample.reshape(28, 28), cmap='RdPu')


# # Confusion Matrix

# In[15]:


confusion_mtx = confusion_matrix(y_true, y_predictions_classes)

# Plot
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap="RdPu")
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix');


# # Investigate  Errors

# In[16]:


errors = (y_predictions_classes - y_true != 0)
y_predictions_classes_errors = y_predictions_classes[errors]
y_predictions_errors = y_predictions[errors]
y_true_errors = y_true[errors]
x_test_errors = x_test[errors]


# In[17]:


y_predictions_errors_probability = np.max(y_predictions_errors, axis=1)
true_probability_errors = np.diagonal(np.take(y_predictions_errors, y_true_errors, axis=1))
diff_errors_predictions_true = y_predictions_errors_probability - true_probability_errors

# Get list of indices of sorted differences
sorted_idx_diff_errors = np.argsort(diff_errors_predictions_true)
top_idx_diff_errors = sorted_idx_diff_errors[-3:] # 3 last ones


# In[18]:


# Show Top Errors
num = len(top_idx_diff_errors)
f, ax = plt.subplots(1, num, figsize=(10,10))

for i in range(0, num):
  idx = top_idx_diff_errors[i]
  sample = x_test_errors[idx].reshape(28,28)
  y_t = y_true_errors[idx]
  y_p = y_predictions_classes_errors[idx]
  ax[i].imshow(sample, cmap='gray')
  ax[i].set_title("Predicted label :{}\nTrue label: {}".format(y_p, y_t), fontsize=22)

