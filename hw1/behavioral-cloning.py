#!/usr/bin/env python
import pickle

import numpy as np
import tensorflow as tf


def model(inp_dim, out_dim):
    inp = tf.placeholder(dtype=tf.float32, shape=[None, inp_dim])
    out = tf.placeholder(dtype=tf.float32, shape=[None, out_dim])

    w0 = tf.get_variable(name='w0', shape=(inp_dim, 20), initializer=tf.contrib.layers.xavier_initializer())
    w1 = tf.get_variable(name='w1', shape=(20, 20), initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable(name='w2', shape=(20, out_dim), initializer=tf.contrib.layers.xavier_initializer())

    b0 = tf.get_variable(name='b0', shape=20, initializer=tf.constant_initializer())
    b1 = tf.get_variable(name='b1', shape=20, initializer=tf.constant_initializer())
    b2 = tf.get_variable(name='b2', shape=out_dim, initializer=tf.constant_initializer())

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


def get_shape(data):
    return np.asarray(data.shape)[1:]


def flattened(shape_arr):
    from functools import reduce
    return reduce(lambda x, y: x * y, shape_arr)


def train(input_data, output_data, save_path):
    inp_shape = get_shape(input_data)
    out_shape = get_shape(output_data)

    inp_dim = flattened(inp_shape)
    out_dim = flattened(out_shape)

    inp, out, pred = model(inp_dim, out_dim)
    mse = tf.reduce_mean(0.5 * tf.square(pred - out))
    opt = tf.train.AdamOptimizer().minimize(mse)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch_size = 32
        for step in range(10000):
            indices = np.random.randint(0, len(input_data), batch_size)
            inp_batch = input_data[indices]
            out_batch = output_data[indices]

            inp_batch = inp_batch.reshape(batch_size, inp_dim)
            out_batch = out_batch.reshape(batch_size, out_dim)

            mse_run, opt_run = sess.run([mse, opt], feed_dict={inp: inp_batch, out: out_batch})

            if step % 1000 == 0:
                print('step: {0:04d}, mse: {1:.3f}, opt: {1:.3f}'.format(step, mse_run, opt_run))
                saver.save(sess, save_path)

        # pred_run = sess.run(pred, feed_dict={inp: inputs})


def inference(gym_env):
    model_file = get_save_path(gym_env)

    import gym
    env = gym.make(gym_env)
    obs = env.reset()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    with tf.Session() as sess:
        inp, out, pred = model(inp_dim=obs_dim, out_dim=action_dim)
        tf.train.Saver().restore(sess, model_file)

        done = False
        totalr = 0.
        steps = 0

        while not done:
            obs = obs.reshape(1, obs_dim)
            action = sess.run(pred, feed_dict={inp: obs})

            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            env.render()
            if steps % 100 == 0: print("%i/%i" % (steps, 1000))
            if steps >= 1000:
                break
        return totalr


def train_save(gym_env):
    save_path = get_save_path(gym_env)
    with open('experts/' + gym_env + '-data.pkl', 'rb') as pickle_file:
        expert_data = pickle.load(pickle_file)
        train(input_data=expert_data['observations'],
              output_data=expert_data['actions'],
              save_path=save_path)


def get_save_path(gym_env):
    save_dir = './model/' + gym_env
    save_path = save_dir + '/model.ckpt'
    import os
    os.makedirs(save_dir, exist_ok=True)
    return save_path


def main():
    gym_env = 'Hopper-v2'
    # train_save(gym_env)

    total_reward = inference(gym_env)
    print('returns', total_reward)


if __name__ == '__main__':
    main()
