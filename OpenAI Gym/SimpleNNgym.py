import tensorflow as tf
import gym
import numpy as np


num_inputs = 4 #Observation inputs
num_hidden = 4
num_outputs = 1 #probability to go left             1-left=right

initializer = tf.contrib.layers.variance_scaling_initializer()

x = tf.placeholder(dtype=tf.float32, shape=[None, num_inputs])

hidden_layer_one = tf.layers.dense(x, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
hidden_layer_two = tf.layers.dense(hidden_layer_one, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
output_layer = tf.layers.dense(hidden_layer_two, num_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)

probabilities = tf.concat(axis=1, values=[output_layer, 1-output_layer])

action = tf.multinomial(probabilities, num_samples=1)

init = tf.global_variables_initializer()

epi = 50     #Run the game 50 times
epochs = 500 #epochs in each epi
avg_steps = []
env = gym.make('CartPole-v0')


with tf.Session() as sess:
    init.run()

    for i_episode in range(epi):
        obs = env.reset()

        for epoch in range(epochs):
            action_val = action.eval(feed_dict={x: obs.reshape(1, num_inputs)})
            obs, reward, done, info = env.step(action_val[0][0]) # 0 or 1

            if done:
                avg_steps.append(epoch)
                print('Done after {} epochs'.format(epoch))
                break

print('After {} episodes, avg_steps per game was {}'.format(epi, np.mean(avg_steps)))

            

