# Creating model architecture.
#from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import layers
from LitRevSentences.constants import *

class ModelArchitecture:

    def __init__(self):
        pass

    
    def get_model(self, text_vectorizer, token_embed):

        inputs = layers.Input(shape=(1,), dtype=tf.string)
        text_vectors = text_vectorizer(inputs)
        token_embeddings = token_embed(text_vectors)
        x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(token_embeddings)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(5, activation="softmax")(x)
        model = tf.keras.Model(inputs,outputs)

        #Compile
        model.compile(loss="categorical_crossentropy",
                        optimizer="adam",
                        metrics=["accuracy"])

        return model