# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import pandas as pd
import numpy as np


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.91 and logs.get('val_acc') > 0.91:
            self.model.stop_training = True


def my_model(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=['acc']
    )
    return model


def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    train_sentence, test_sentence, train_labels, test_labels = train_test_split(
        bbc.text,
        bbc.category,
        train_size=training_portion,
        shuffle=False
    )
    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentence)

    padding = lambda sentence: \
        tf.keras.preprocessing.sequence.pad_sequences(
            tokenizer.texts_to_sequences(sentence),
            maxlen=max_length,
            truncating=trunc_type,
            padding=padding_type
        )

    train_padded, test_padded = padding(train_sentence), padding(test_sentence)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(bbc.category)

    train_labels_final = np.array(label_tokenizer.texts_to_sequences(train_labels))
    test_labels_final = np.array(label_tokenizer.texts_to_sequences(test_labels))

    model = my_model(vocab_size, embedding_dim, max_length)

    model.fit(
        train_padded,
        train_labels_final,
        epochs=100,
        validation_data=(
            test_padded,
            test_labels_final
        ),
        callbacks = myCallback()
    )
    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.


if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
