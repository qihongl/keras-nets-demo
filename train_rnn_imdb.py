'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
reference: https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
'''

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
# from keras.datasets import imdb
# from keras.preprocessing import sequence
from data_loader import load_imdb
from models import get_simple_lstm
import numpy as np
import os

# log_root = '/tigress/qlu/logs/keras-resnet/log'
log_root = 'log'
data_name = 'imdb'
model_name = 'lstm1'
max_epoches = 10
# nsubjs = 10

# load IMDB
batch_size = 64
max_features = 20000
max_len = 80
# load data
x_train, y_train, x_test, y_test = load_imdb(max_features, max_len)

# create a log dir
subj_id = 0
# for subj_id in range(nsubjs):
log_dir = os.path.join(log_root, data_name, model_name, 'subj%.2d' % (subj_id))
print(log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# create various callbacks
lr_reducer = ReduceLROnPlateau(
    factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-5)
csv_logger = CSVLogger(os.path.join(log_dir, 'history.csv'))
checkpointer = ModelCheckpoint(
    filepath=os.path.join(log_dir, 'weights.{epoch:03d}.hdf5'),
    verbose=1, save_best_only=False, period=1)

# build the model
model = get_simple_lstm(max_features)
# save initial weights
model.save_weights(os.path.join(log_dir, 'weights.%.3d.hdf5' % (0)))

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=max_epoches,
          validation_data=(x_test, y_test),
          callbacks=[lr_reducer, csv_logger, checkpointer],
          verbose=1)
