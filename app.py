import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('saved_models/clasificador-iris')

##################################################################
#Datos de prueba para predecir
new_items = []

fil2 = [[4.3,6.2,3.2,0.4],[7.2,4.1,6.2,0.2],[7.3,6.2,1.5,2],[6.5,4.2,5.5,1.7]]

for item in fil2:
  step = np.array(item).reshape([2,2])
  new_items.append(step)

test = np.array(new_items)
test = np.uint8(test)

####################################################################

predictions = model.predict(test)

results = []

for i in predictions:
    result = np.argmax(i)
    results.append(result)

print(f"Los resultados son: {results}")