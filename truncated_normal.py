import tensorflow as tf


if __name__ == '__main__':
    # 这是一个截断的产生正太分布的函数，
    # 就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
    # 和一般的正太分布的产生随机数据比起来，
    # 这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
    c = tf.truncated_normal(shape=[2, 1], mean=0, stddev=0.1)
    b = tf.zeros([3, 8])
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print(sess.run(c))
        print(sess.run(b))
