import tensorflow as tf
from tensorflow.contrib import layers as layers


class DNN(object):
  def create_model(self, model_inputs):
    initializer = tf.random_normal
    w1 = tf.Variable(initializer(shape=[784, 60]))
    b1 = tf.Variable(initializer(shape=[60]))

    w2 = tf.Variable(initializer(shape=[60, 30]))
    b2 = tf.Variable(initializer(shape=[30]))
 
    w3 = tf.Variable(initializer(shape=[30, 10]))
    b3 = tf.Variable(initializer(shape=[10]))

    h1 = tf.nn.relu(tf.matmul(model_inputs, w1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    logits = tf.matmul(h2, w3) + b3
    predictions = tf.nn.softmax(logits)

    return predictions

class contrib_DNN(object):
  def create_model(self, model_inputs):
    h1 = layers.fully_connected(
      inputs=model_inputs,
      num_outputs=60,
      activation_fn=tf.nn.relu)

    h2 = layers.fully_connected(
      inputs=h1,
      num_outputs=30,
      activation_fn=tf.nn.relu)

    logits = layers.fully_connected(
      inputs=h2,
      num_outputs=10)

    predictions = tf.nn.softmax(logits)

    return predictions



    

