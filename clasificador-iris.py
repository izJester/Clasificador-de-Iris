import tensorflow as tf
import pandas as pd
import numpy as np


iris_data = pd.read_csv("./iris-data.csv")

train_labels = np.array(iris_data['species'])


for i in range(len(iris_data)):
    if train_labels[i] == 'setosa':
        train_labels[i] = 0
    elif train_labels[i] == 'versicolor':
        train_labels[i] = 1
    elif train_labels[i] == 'virginica':
        train_labels[i] = 2
        
values = iris_data.loc[ : , iris_data.columns != 'species']

fil = values.values

new_values = []

for i in fil:
    value = np.array(i).reshape(2,2)
    new_values.append(value)

train = np.array(new_values)

train = np.uint8(train)
train_labels = np.uint8(train_labels)

labels = ['setosa' , 'versicolor' , 'virginica']


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(2,2)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train, train_labels, epochs=250)

test_loss, test_acc = model.evaluate(train,  train_labels, verbose=2)
print(f"Presición del {test_acc * 100}%")

model.save('saved_models/clasificador-iris')



