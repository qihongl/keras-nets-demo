import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, LSTM


def get_mnist_net(model_name, input_shape):
    if model_name == 'std':
        return get_mnist_stdnet()
    elif model_name == 'conv':
        return get_mnist_convnet(input_shape)
    else:
        raise(':(')


def get_mnist_convnet(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def get_mnist_stdnet():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def get_simple_lstm(max_features):
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
