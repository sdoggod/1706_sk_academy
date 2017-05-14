import tensorflow as tf
import numpy as np
import models
from tensorflow.examples.tutorials.mnist import input_data
from IPython import embed
from tensorflow import flags
FLAGS = flags.FLAGS
flags.DEFINE_string("log_dir", "./logs/default", "default summary/checkpoint directory")
flags.DEFINE_float("learning_rate", 0.01, "base learning rate")
flags.DEFINE_string("model", "DNN", "model name")
flags.DEFINE_string("optimizer", "GradientDescentOptimizer", "kind of optimizer to use.")
flags.DEFINE_integer("batch_size", 1024, "default batch size.")
flags.DEFINE_integer("max_steps", 10000, "number of max iteration to train.")


def main(_):
  mnist = input_data.read_data_sets("./data", one_hot=True)

  # defien model input: image and ground-truth label
  model_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784])
  labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

  model = getattr(models, FLAGS.model, None)()
  predictions = model.create_model(model_inputs)

  # define cross entropy loss term
  loss = tf.losses.softmax_cross_entropy(
    onehot_labels=labels,
    logits=predictions)

  dense_predictions = tf.argmax(predictions, axis=1)
  dense_labels = tf.argmax(labels, axis=1)
  equals = tf.cast(tf.equal(dense_predictions, dense_labels), tf.float32)
  acc = tf.reduce_mean(equals)

  tf.summary.scalar("loss", loss)
  tf.summary.scalar("acc", acc)
  merge_op = tf.summary.merge_all()

  optimizer = getattr(tf.train, FLAGS.optimizer, None)(FLAGS.learning_rate)
  train_op = optimizer.minimize(loss)

  saver = tf.train.Saver()

  with tf.Session() as sess:
    summary_writer_train = tf.summary.FileWriter(FLAGS.log_dir + "/train", sess.graph)
    summary_writer_val = tf.summary.FileWriter(FLAGS.log_dir + "/validation", sess.graph)

    sess.run(tf.global_variables_initializer())
    for step in range(FLAGS.max_steps):
      batch_images, batch_labels = mnist.train.next_batch(FLAGS.batch_size)
      images_val, labels_val = mnist.validation.next_batch(FLAGS.batch_size)
      feed = {model_inputs: batch_images, labels: batch_labels}
      _, loss_val = sess.run([train_op, loss], feed_dict=feed)
      print "step {} | loss {}".format(step, loss_val)
      if step % 10 == 0:
        summary_train = sess.run(merge_op, feed_dict=feed)
        feed = {model_inputs: images_val, labels:labels_val}
        summary_val = sess.run(merge_op, feed_dict=feed)
        summary_writer_train.add_summary(summary_train, step)
        summary_writer_val.add_summary(summary_val, step)

      if step % 1000 == 0:
        save_path = saver.save(sess, FLAGS.log_dir + "/model.ckpt", global_step=step)
        print "step {} | model saved at {}".format(step, save_path)

if __name__ == "__main__":
  tf.app.run()




