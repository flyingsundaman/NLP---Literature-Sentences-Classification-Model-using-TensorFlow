# Creating model architecture.
from LitRevSentences.entity.config_entity import ModelTrainerConfig
#from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D,LSTM,Activation,Dense,Dropout,Input,Embedding,SpatialDropout1D
from tensorflow.python.eager.monitoring import Metric
from LitRevSentences.constants import *

class ModelArchitecture:

    def __init__(self):
        pass

    
    def get_model(self, text_vectorizer, token_embed):
        """model = Sequential()
        model.add(Embedding(MAX_WORDS, 100,input_length=MAX_LEN))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(1,activation=ACTIVATION))
        model.summary()
        model.compile(loss=LOSS,optimizer=RMSprop(),metrics=METRICS)"""

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