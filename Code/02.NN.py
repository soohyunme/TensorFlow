import os
from threading import active_count

from tensorflow.python.keras.layers.core import Activation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape x
x_train = x_train.reshape(-1,28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1,28*28).astype("float32") / 255.0

# Non nomalize
x_train = x_train.reshape(-1,28*28).astype("float32") 
x_test = x_test.reshape(-1,28*28).astype("float32") 

# Sequential API (Very convenient, not very flexible)
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10), # loss(from_logits=True)
    ]
)

model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu', name='my_layer'))
model.add(layers.Dense(10)) # loss(from_logits=True) 

# model = keras.Model(inputs=model.inputs, 
#                     outputs=[layer.output for layer in model.layers]) 
#                     # outputs=[model.get_layer('my_layer').output])
#                     # outputs=[model.layers[-2].output])


# Functional API (A bit more flexible)
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu', name='second_layer')(x)
x = layers.Dense(128, activation='relu', name='third_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x) # loss(from_logits=False)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    # optimizer = keras.optimizers.SGD(lr=0.001), # Try SGD
    # optimizer = keras.optimizers.Adagrad(lr=0.001), # Try Adagrad
    # optimizer = keras.optimizers.RMSprop(lr=0.001), # Try RMSprop
    metrics=["accuracy"],
)


model.fit(x_train, y_train, batch_size=64, epochs=15, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)

'''
SUGGESTIONS:
1. Try and see what accuracy you can get by increasing the model, training for longer, etcetera.
You should be able to get over 98.2% on the test set!
Baseline = 0.9779
-> Add layer(1 layer) and more epochs(5->10) 0.9813
-> bigger batch size 32 -> 64 and more epochs(10->15) 0.9828

2. Try using different optimizers than Adam, 
for example Gradient Descent with Momentum, Adagrad, and RMSprop

Use SGD in last model -> 0.9225
Use Adagrad in last model -> 0.9401
Use RMSprop in last model -> 0.9799

3. Is there any difference if you remove the normalization of the data?
normalize -> 0.9828
Non normalize -> 0.9809

'''
