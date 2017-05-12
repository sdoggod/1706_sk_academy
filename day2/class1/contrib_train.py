import tensorflow as tf
import numpy as np
import models
from tensorflow.examples.tutorials.mnist import input_data
from IPython import embed
from tensorflow import flags
FLAGS = flags.FLAGS
flags.DEFINE_string("log_dir", "./logs/default", "default summary/checkpoint directory")
def main(_):
  mnist = input_data.read_data_sets("./data", one_hot=True)

  # defien model input: image and ground-truth label
  model_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784])
  labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=784)]

  classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 20, 10],
    n_classes=10,
    model_dir="./logs")

  def get_train_inputs():
    x, y = mnist.train.next_batch(128)
    x = tf.constant(x)
    y = tf.constant(y)
    y = tf.cast(tf.argmax(y, 1), tf.int64)
    return x, y

  classifier.fit(input_fn=get_train_inputs, steps=2000)
  score = classifier.evaluate(input_fn=get_train_inputs, steps=1000)["accuracy"]
  print "accuracy : {}".format(score)

if __name__ == "__main__":
  tf.app.run()




