"""
Reference: modify pretarined network
https://stackoverflow.com/questions/42475381/add-dropout-layers-between-pretrained-dense-layers-in-keras
"""
from keras.applications import VGG16
from keras.layers import Dropout, GaussianNoise
from keras.models import Model

model = VGG16(weights='imagenet')

# Store the fully connected layers
fc1 = model.layers[-3]
fc2 = model.layers[-2]
predictions = model.layers[-1]

# Create the dropout layers
stddev_val = 1
noise_layer = GaussianNoise(stddev_val)

# Reconnect the layers
x = fc1.output
x = noise_layer(x)
x = fc2(x)
predictors = predictions(x)

# Create a new model
model2 = Model(inputs=model.input, outputs=predictors)
model2.summary()
model.summary()
