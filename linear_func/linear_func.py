import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers



inputs = layers.Input(shape=(1,))
outputs = layers.Dense(1, activation="linear")(inputs)
model = keras.Model(inputs=inputs, outputs=outputs)
# Alternative init:
# model = keras.Sequential()
# model.add(layers.Dense(units=1, input_shape=(1,), activation='linear'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))


x = np.array(list(range(-100, 100, 10)))
y = np.array([it*10 - 3 for it in x])
history = model.fit(x, y, epochs=500, verbose=0)


print(model.predict([33]))
print(model.get_weights())


plt.plot(history.history['loss'])
plt.grid(True)
plt.show()
