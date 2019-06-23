import keras
import matplotlib.pyplot as plt
import numpy as np
import pylab
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential


def gen_data():
    # generate the data
    inputs = np.linspace(-2 * np.pi, 2 * np.pi, 10000)[:, None]
    outputs = np.sin(inputs) + 0.05 * np.random.normal(size=[len(inputs), 1])

    plt.scatter(inputs[:, 0], outputs[:, 0], s=0.1, color='k', marker='o')
    pylab.show()
    return inputs, outputs


def kerasModel():
    model = Sequential()

    model.add(Dense(units=20,
                    input_dim=1,
                    activation='relu',
                    kernel_initializer='glorot_uniform',
                    use_bias=True,
                    bias_initializer='constant'
                    ))
    model.add(Dense(units=20,
                    activation='relu',
                    kernel_initializer='glorot_uniform',
                    use_bias=True,
                    bias_initializer='constant'
                    ))
    model.add(Dense(units=1,
                    kernel_initializer='glorot_uniform',
                    use_bias=True,
                    bias_initializer='constant'
                    ))

    model.summary()
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='adam',
                  metrics=['acc'])

    inputs, outputs = gen_data()
    model.fit(x=inputs, y=outputs, batch_size=32, epochs=30)

    pred = model.predict(x=inputs)

    plt.scatter(inputs[:, 0], outputs[:, 0], c='k', marker='o', s=0.1)
    plt.scatter(inputs[:, 0], pred[:, 0], c='b', marker='o', s=0.1)
    pylab.show()



def model():
    inp = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    out = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    w0 = tf.get_variable(name='w0', shape=(1, 20), initializer=tf.contrib.layers.xavier_initializer())
    w1 = tf.get_variable(name='w1', shape=(20, 20), initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable(name='w2', shape=(20, 1), initializer=tf.contrib.layers.xavier_initializer())

    b0 = tf.get_variable(name='b0', shape=20, initializer=tf.constant_initializer())
    b1 = tf.get_variable(name='b1', shape=20, initializer=tf.constant_initializer())
    b2 = tf.get_variable(name='b2', shape=1, initializer=tf.constant_initializer())

    weights = [w0, w1, w2]
    biases = [b0, b1, b2]
    activations = [tf.nn.relu, tf.nn.relu, None]

    layer = inp
    for w, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, w) + b
        if activation is not None:
            layer = activation(layer)

    pred = layer
    return inp, out, pred


def train_and_infer():
    inputs, outputs = gen_data()

    inp, out, pred = model()
    mse = tf.reduce_mean(0.5 * tf.square(pred - out))
    opt = tf.train.AdamOptimizer().minimize(mse)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch_size = 32
        for step in range(10000):
            indices = np.random.randint(0, len(inputs), batch_size)
            inp_batch = inputs[indices]
            out_batch = outputs[indices]

            mse_run, opt_run = sess.run([mse, opt], feed_dict={inp: inp_batch, out: out_batch})

            if step % 1000 == 0:
                print('step: {0:04d}, mse: {1:.3f}, opt: {1:.3f}'.format(step, mse_run, opt_run))

        pred_run = sess.run(pred, feed_dict={inp: inputs})

    plt.scatter(inputs[:, 0], outputs[:, 0], c='k', marker='o', s=0.1)
    plt.scatter(inputs[:, 0], pred_run[:, 0], c='r', marker='o', s=0.1)
    pylab.show()


def main():
    # train_and_infer()
    kerasModel()


if __name__ == '__main__':
    main()
