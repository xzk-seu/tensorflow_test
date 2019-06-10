import tensorflow as tf
from tensorflow.contrib import rnn, layers
from tqdm import tqdm

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
# 矩阵分解
x1 = tf.unstack(x, n_steps, 1)
# lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
# outputs, states = rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)

fw_cells = [rnn.LSTMCell(n_hidden), rnn.LSTMCell(n_hidden)]
bw_cells = [rnn.LSTMCell(n_hidden), rnn.LSTMCell(n_hidden)]

'''
output是一个tuple(fw_output, bw_output)
其中每个元素为[batch_size, max_time, layer_output]
'''
outputs, _, _ = rnn.stack_bidirectional_rnn(fw_cells, bw_cells, x1, dtype=tf.float32)

pred = layers.fully_connected(outputs[-1], n_classes, activation_fn=None)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
model_path = 'bi_log/bi_lstm.ckpt'

batch_size = 100

config = tf.ConfigProto(allow_soft_placement=True)
# 最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # 共55000个训练样例
    training_epochs = 10
    for epoch in tqdm(range(training_epochs)):
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape(batch_size, -1, 28)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        # print("Iter%d, step %d, training accuracy %g" % (mnist.train.epochs_completed, (i + 1), train_accuracy))

    # 计算测试数据的准确率
    temp_x = mnist.test.images.reshape(-1, 28, 28)
    print("test accuracy %g" % sess.run(accuracy, feed_dict={x: temp_x, y: mnist.test.labels}))
    save_path = saver.save(sess, model_path)
    print('saving....')
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver.restore(sess, model_path)
#     correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#     print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
#
#     output = tf.argmax(pred, axis=1)
#
#     batch_xs, batch_ys = mnist.train.next_batch(2)
#