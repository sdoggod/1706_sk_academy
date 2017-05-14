import tensorflow as tf
import numpy as np
import models
from tensorflow.examples.tutorials.mnist import input_data
from IPython import embed
from tensorflow import flags
FLAGS = flags.FLAGS
flags.DEFINE_string("log_dir", "./logs/default", "default summary/checkpoint directory")
flags.DEFINE_string("model", "DNN", "model name")
flags.DEFINE_integer("batch_size", 100, "default batch size.")
flags.DEFINE_integer("max_steps", 50, "number of max iteration to train.")

def main(_):
  mnist = input_data.read_data_sets("./data", one_hot=True)

  # defien model input: image and ground-truth label
  model_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784])
  labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

  model = getattr(models, FLAGS.model, None)()
  predictions = model.create_model(model_inputs)

  dense_predictions = tf.argmax(predictions, axis=1)
  dense_labels = tf.argmax(labels, axis=1)
  equals = tf.cast(tf.equal(dense_predictions, dense_labels), tf.float32)
  acc = tf.reduce_mean(equals)

  saver = tf.train.Saver()
  final_acc = 0.0
  with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.latest_checkpoint(FLAGS.log_dir)
    print checkpoint
    saver.restore(sess, checkpoint)
    for step in range(FLAGS.max_steps):
      images_val, labels_val = mnist.validation.next_batch(FLAGS.batch_size)
      feed = {model_inputs: images_val, labels:labels_val}
      acc_value = sess.run(acc, feed_dict=feed)
      final_acc += acc_value
    final_acc /= float(FLAGS.max_steps)
    print "Full Validation Accuracy : {}".format(final_acc)

if __name__ == "__main__":
  tf.app.run()




