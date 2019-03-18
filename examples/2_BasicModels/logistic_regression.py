'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
#%%
from __future__ import print_function

import tensorflow as tf

# Import MNIST data
# 本例使用MNIST，数字识别数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
# 数据集每个图片是28*28像素，结果是0-9共10个数字
# None保留，可以是任何数，在这一例中后续作为每批测试数据数量(batch_size)输入
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
# 创建变量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
# 回归方程 Y = x·W + b ， 可判断出Y为shape为(None, 10)的张量
# 预测函数 pred = softmax(Y) ，对Y应用归一化指数函数，使输出概率化，
# 得到的结果看作0-9数字的预测概率，取最大值则为预测结果。
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
# 使用交叉熵作代价函数
# 知y(None,10),,pred(None,10), y*tf.log(pred)对应位相乘得到(None,10)的张量；
# reduce_sum对维度alix=1求和降维，效果就是sum(p*log(1/q))，得到(None)个交叉熵，
# 即每一个单独训练集的交叉熵。
# reduce_mean对(None)个交叉熵求平均值，得到这一次训练的平均交叉熵，并以此为代价进行优化。
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
# 对代价函数cost使用梯度下降优化W,b变量
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    # 进行training_epochs次训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        # 每次取batch_size条数据，共需total_batch次
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            # 执行训练，并记录每次训练得到的cost以计算整个训练集的平均cost
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss 平均cost
            avg_cost += c / total_batch

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    # argmax返回pred在维度alis=1上最大值的下标，pred是(None,10)，所以得到(None)个预测值（概率最高）
    # correct_prediction就是(None)个bool，是否预测正确
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy 准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 将MNIST的测试数据代入，计算准确率
    # .eval 与 sess.run(accuracy)相同
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
