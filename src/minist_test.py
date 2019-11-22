from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf



def main():

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    sess.run(tf.global_variables_initializer())

    y = tf.matmul(x, W) + b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    return



if __name__ == '__main__':
    main()