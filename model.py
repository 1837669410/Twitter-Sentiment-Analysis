import tensorflow as tf
from tensorflow import keras
from data import load_data
from utils import set_soft_gpu

class TextCNN(keras.Model):

    def __init__(self, vocab_num, out_dim, units, dropout_rate, l_rate, decay):
        super(TextCNN, self).__init__()

        # embedding
        self.embedding = keras.layers.Embedding(input_dim=vocab_num, output_dim=out_dim, mask_zero=True)
        # conv
        self.conv1 = [keras.layers.Conv2D(filters=units, kernel_size=[i,out_dim], strides=1, padding="valid", activation="relu") for i in [2,3,4]]
        self.conv2 = [keras.layers.Conv2D(filters=units, kernel_size=[i,out_dim], strides=1, padding="valid", activation="relu") for i in [2,3,4]]
        # pool
        self.pool1 = [keras.layers.MaxPool2D(pool_size=[i,1], strides=1, padding="valid") for i in [19,18,17]]
        self.pool2 = [keras.layers.MaxPool2D(pool_size=[i,1], strides=1, padding="valid") for i in [19,18,17]]
        # dropout
        self.dropout = keras.layers.Dropout(dropout_rate)
        # dense
        self.dense1 = keras.layers.Dense(100)
        self.dense2 = keras.layers.Dense(4)
        # relu
        self.relu = keras.layers.ReLU()

        # loss_func
        self.loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)
        # opt
        self.opt = keras.optimizers.Adam(l_rate, decay=decay)

    def call(self, inputs, training=None, mask=None):
        # embedding
        emb = self.embedding(inputs)
        emb = tf.expand_dims(emb, axis=3)
        # conv
        out11, out12, out13 = self.conv1[0](emb), self.conv1[1](emb), self.conv1[2](emb)
        out21, out22, out23 = self.conv2[0](emb), self.conv2[1](emb), self.conv2[2](emb)
        # pool
        out11, out12, out13 = self.pool1[0](out11), self.pool1[1](out12), self.pool1[2](out13)
        out21, out22, out23 = self.pool2[0](out21), self.pool2[1](out22), self.pool2[2](out23)
        # concat
        out = tf.concat((out11, out12, out13, out21, out22, out23), axis=3)
        out = tf.squeeze(out)
        # dense
        out = self.dense1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.dense2(out)
        return out

def train():
    epoch = 10
    set_soft_gpu(True)
    db_train, db_val, vocab_num = load_data("twitter_training.csv", "twitter_validation.csv", max_text_length=20, framework="tf")
    model = TextCNN(vocab_num=vocab_num, out_dim=64, units=100, dropout_rate=0.5, l_rate=0.0003, decay=0.001)
    model.build(input_shape=[128, 20])
    model.summary()
    for e in range(epoch):
        for step, (x,y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                out = model.call(x)
                y = tf.one_hot(y, depth=4)
                loss = model.loss_func(y, out)
                grads = tape.gradient(loss, model.trainable_variables)
            model.opt.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print("epoch: %d | step: %d | epoch:%.3f"%(e,step,loss))

        total_acc = 0
        total_num = 0
        for step, (x,y) in enumerate(db_val):
            out = model.call(x)
            pred = tf.nn.softmax(out, axis=1)
            pred = tf.cast(tf.argmax(pred, axis=1), dtype=tf.int32)
            pred = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            total_acc += tf.reduce_sum(pred)
            total_num += x.shape[0]
        print("epoch: %d | acc:%.3f"%(e, total_acc / total_num))

    model.save_weights("./model/text_cnn.ckpt")

if __name__ == "__main__":
    train()