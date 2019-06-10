import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
tf.reset_default_graph()

# 用动态RNN处理变长序列，
# 两个序列，长度分别为4，1
X = np.random.randn(2, 4, 5)
# X = [[[.....],[......],[......],[......]],
#      [[.....],[......],[......],[......]]]
X[1, 1:] = 0
# X = [[[.....],[......],[......],[......]],
#      [[.....],[00000],[00000],[00000]]]
seq_lengths = [4, 1]
cell = rnn.BasicLSTMCell(num_units=3, state_is_tuple=True)
# Gate Recurrent Unit
gru = rnn.GRUCell(num_units=3)

outputs, last_states = tf.nn.dynamic_rnn(cell, X, seq_lengths, dtype=tf.float64)

gruoutputs, grulast_states = tf.nn.dynamic_rnn(gru, X, seq_lengths, dtype=tf.float64)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
result, sta, gruout, grusta = sess.run([outputs, last_states, gruoutputs, grulast_states])
print('全序列： \n', result[0])
print('短序列： \n', result[1])
print('LSTM状态: ', len(sta), '\n', sta[1])
print('GRU短序列： \n', gruout[1])
print('LSTM状态: ', len(grusta), '\n', grusta[1])
