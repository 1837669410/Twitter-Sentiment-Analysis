import tensorflow as tf
from tensorflow import keras

# Input
input = keras.Input(shape=(20,), dtype='float32', name='input')
# embedding
emb = keras.layers.Embedding(input_dim=25041, output_dim=64)(input)
emb = tf.expand_dims(emb, axis=3)
# conv
out11 = keras.layers.Conv2D(filters=100, kernel_size=[2,64], strides=1, padding="valid", activation="relu")(emb)
out12 = keras.layers.Conv2D(filters=100, kernel_size=[3,64], strides=1, padding="valid", activation="relu")(emb)
out13 = keras.layers.Conv2D(filters=100, kernel_size=[4,64], strides=1, padding="valid", activation="relu")(emb)
out21 = keras.layers.Conv2D(filters=100, kernel_size=[2,64], strides=1, padding="valid", activation="relu")(emb)
out22 = keras.layers.Conv2D(filters=100, kernel_size=[3,64], strides=1, padding="valid", activation="relu")(emb)
out23 = keras.layers.Conv2D(filters=100, kernel_size=[4,64], strides=1, padding="valid", activation="relu")(emb)
# pool
out11 = keras.layers.MaxPool2D(pool_size=[19,1], strides=1, padding="valid")(out11)
out12 = keras.layers.MaxPool2D(pool_size=[18,1], strides=1, padding="valid")(out12)
out13 = keras.layers.MaxPool2D(pool_size=[17,1], strides=1, padding="valid")(out13)
out21 = keras.layers.MaxPool2D(pool_size=[19,1], strides=1, padding="valid")(out21)
out22 = keras.layers.MaxPool2D(pool_size=[18,1], strides=1, padding="valid")(out22)
out23 = keras.layers.MaxPool2D(pool_size=[17,1], strides=1, padding="valid")(out23)
# concat
out = tf.concat((out11, out12, out13, out21, out22, out23), axis=3)
out = keras.layers.Flatten()(out)
# dense
out = keras.layers.Dense(100)(out)
out = keras.layers.ReLU()(out)
out = keras.layers.Dropout(0.5)(out)
out = keras.layers.Dense(4)(out)
model = tf.keras.Model(inputs=input, outputs=[out])
tf.keras.utils.plot_model(model, show_shapes=True, to_file="./static/model.png")