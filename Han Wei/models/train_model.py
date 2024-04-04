from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

from data.make_dataset import IMAGE_SIZE

def create_model(input_shape):
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten layer
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))

    # Output layer
    model.add(Dense(5, activation='linear'))  # 5 neurons for mass, calories, fats, protein, carbs

    return model

def multi_loss(actual, pred):
    pred = tf.cast(pred, dtype=actual.dtype)
    temp = tf.abs(actual - pred)
    l_multi = tf.reduce_mean(tf.reduce_sum(temp[:, 2:], axis=1))
    l = tf.reduce_mean(tf.reduce_sum(temp[:, :2], axis=1) + l_multi)
    return l


def train(train, validate, epochs, batch_size, optimizer="adam", loss="mse"):
    model = create_model(IMAGE_SIZE)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae', loss])
    X_train, y_train = train
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validate)
    return model