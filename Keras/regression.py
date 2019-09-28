import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def convert(input):
    F=model.predict([input])
    return F


tf.logging.set_verbosity(tf.logging.ERROR)
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38, 40, 55, 63, 70],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46.4, 59, 71.6, 100.4, 104, 131, 145.4, 158],  dtype=float)

l0=tf.keras.layers.Dense(units=1, input_shape=[1])
#l1=tf.keras.layers.Dense(units=1)
#l2=tf.keras.layers.Dense(units=1)

model=tf.keras.Sequential([
    l0,
])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.9))

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])


print(convert(int(input("Enter Celcius to convert into Farenheit:\n>>>"))))
print("Learning Graph:")
plt.show()
print(l0.get_weights())
print(l1.get_weights())
print(l2.get_weights())
