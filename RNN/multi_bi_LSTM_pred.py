import tensorflow as tf
from tensorflow.contrib import rnn, layers
import pylab
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
# 矩阵分解
x1 = tf.unstack(x, n_steps, 1)
fw_cells = [rnn.LSTMCell(n_hidden), rnn.LSTMCell(n_hidden)]
bw_cells = [rnn.LSTMCell(n_hidden), rnn.LSTMCell(n_hidden)]

'''
output是一个tuple(fw_output, bw_output)
其中每个元素为[batch_size, max_time, layer_output]
'''
outputs, _, _ = rnn.stack_bidirectional_rnn(fw_cells, bw_cells, x1, dtype=tf.float32)

pred = layers.fully_connected(outputs[-1], n_classes, activation_fn=None)

saver = tf.train.Saver()
model_path = 'bi_log/bi_lstm.ckpt'

batch_size = 100

config = tf.ConfigProto(allow_soft_placement=True)
# 最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)

    correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    x_val = mnist.test.images.reshape(mnist.test.num_examples, -1, 28)
    print('Accuracy:', accuracy.eval({x: x_val, y: mnist.test.labels}))

    output = tf.argmax(pred, axis=1)

    batch_xs, batch_ys = mnist.train.next_batch(2)
    batch_xs = batch_xs.reshape(2, -1, 28)
    outputval, predv = sess.run([output, pred], feed_dict={x: batch_xs})
    print(outputval, predv, batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
