import tensorflow as tf
import numpy as np
from tqdm import tqdm


learning_rate = 1e-4
n_input = 2
n_label = 1
n_hidden = 20

# tf.placeholder(dtype, shape)
# dtype 数据类型
# shape 数据形状 [a, b]表示a行b列， None表示不确定
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_label])

# h1 h2分别代表隐藏层和输出层
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden, n_label], stddev=0.1))
}
biases = {
    'h1': tf.Variable(tf.zeros([n_hidden])),
    'h2': tf.Variable(tf.zeros([n_label]))
}

layer_1 = tf.nn.softmax(tf.add(tf.matmul(x, weights['h1']), biases['h1']))
y_pred = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['h2']))
final_y = tf.round(y_pred)

loss = tf.reduce_mean((y_pred-y)**2)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# 生成数据
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
X = np.array(X).astype('float32')
Y = np.array(Y).astype('int16')

# 加载session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
g = tf.get_default_graph()
for i in tqdm(range(100000)):
    sess.run(train_step, feed_dict={x: X, y: Y})
    if i % 1000 == 0:
        print(i, sess.run(loss, feed_dict={x: X, y: Y}))

print(sess.run(final_y, feed_dict={x: X}))
