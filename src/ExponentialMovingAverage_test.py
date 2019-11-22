import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)
ema = tf.train.ExponentialMovingAverage(decay=0.9)
m = ema.apply([w])
av = ema.average(w)

x = tf.placeholder(tf.float32, [None])
y = tf.placeholder(tf.float32, [None])
y_ = tf.multiply(x, w)

loss = tf.reduce_sum(tf.square(tf.subtract(y, y_)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
opt_op = optimizer.minimize(loss)

with tf.control_dependencies([opt_op]):
    train = tf.group(m, av)

W = []
AV = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _, w_, av_ = sess.run([train, w, av], feed_dict={x: [1], y: [10]})
        #_, w_, av_ = sess.run([opt_op, w, av], feed_dict={x: [1], y: [10]})
        W.append(w_)
        AV.append(av_)

    for i in range(20):
        print(W[i], ',', AV[i])