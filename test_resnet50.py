# reference1: https://github.com/fchollet/deep-learning-models
# reference2: https://github.com/philipperemy/keras-visualize-activations
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from dep.imagenet_utils import preprocess_input, decode_predictions
from dep.read_acts_keras import get_activations
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

# get the model
model = ResNet50(include_top=True, weights='imagenet',
                 input_tensor=None, input_shape=None,
                 pooling=None, classes=1000)

# load an image
img_dir = 'imgs'
img_name = 'stanford.jpg'
img_path = join(img_dir, img_name)
img = image.load_img(img_path, target_size=(224, 224))
# plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# feed the image
preds = model.predict(x)
print(decode_predictions(preds))

# fetch the activities
activations = get_activations(model, x, print_shape_only=True)
len(activations)
