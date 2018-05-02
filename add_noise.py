"""
Reference: modify pretarined network
https://stackoverflow.com/questions/42475381/add-dropout-layers-between-pretrained-dense-layers-in-keras
"""
from keras.applications import VGG16
from keras.layers import GaussianNoise, Dense
from keras.models import Model
from keras.preprocessing import image
# from keras.utils import plot_model
from os.path import join
import numpy as np
from dep.read_acts_keras import get_activations
from dep.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=1.3)
model = VGG16(weights='imagenet')

# Store the fully connected layers
last_hidden_layer = model.layers[-2]
output_layer = model.layers[-1]
output_weights = output_layer.get_weights()

# Reconnect the layers
x = last_hidden_layer.output
stddev_val = 3
x = GaussianNoise(stddev_val)(x)
new_output = Dense(output_layer.get_config()['units'],
                   activation=output_layer.get_config()['activation'],
                   name='new_output')(x)

# Create a new model
model2 = Model(inputs=model.input, outputs=new_output)
model2.layers[-1].set_weights(output_weights)
# model2.summary()
# model.summary()
# plot_model(model2, to_file='model.png')


# load an image
img_dir = 'imgs'
img_name = 'stanford.jpg'
img_path = join(img_dir, img_name)
img = image.load_img(img_path, target_size=(224, 224))
# plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

acts_train = get_activations(
    model2, x, testing_mode=False,
    print_shape_only=True, layer_name='new_output')

acts_test = get_activations(
    model2, x, testing_mode=True,
    print_shape_only=True, layer_name='new_output')


plt.plot(np.squeeze(acts_train), label='noisy')
plt.plot(np.squeeze(acts_test), label='actual')
plt.legend()
