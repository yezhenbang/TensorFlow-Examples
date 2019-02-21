'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
#%%
from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt 
rng = numpy.random

# Parameters
learning_rate = 0.02
training_epochs = 1500
display_step = 100

# Training Data
# asarray：转换为ndarray对象
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# tf Graph Input
# 声明float32类型，未定义形状（shape）
X = tf.placeholder("float")
Y = tf.placeholder("float")
#print(X)

# Set model weights
# 创建变量W和b，作为线性方程的参数，并初始化为float随机数
W = tf.Variable(rng.randn(), name="weight", trainable=True)
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
# 线性模型 pred = WX + b
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
# 均方误差 cost = E(pred - Y)^2 表示梯度下降中的代价函数，值越小表示越拟合数据
# 1/2系数使得平方求梯度后常数系数为1，方便计算，系数对结果不影响
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
# 使用梯度下降算法模型优化器。minimize(cost)包括compute_gradients(cost)和apply_gradients()。
# compute_gradients(cost)计算cost的梯度，默认使用GraphKeys.TRAINABLE_VARIABLES，所以包括了变量W和b
# apply_gradients()应用梯度到变量列表，变量W和b为trainable，在这一步更新变量W和b的值
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
# 初始化变量的操作
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    # 初始化，初始化变量
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs): #训练次数
        # for (x, y) in zip(train_X, train_Y): #将X,Y打包成(xi,yi)对的形式
            # sess.run(optimizer, feed_dict={X: x, Y: y}) #将每组(xi,yi)代入优化器计算

        # 用下面这句替代上面一层循环，得到接近的结果，但速度明显提升，为什么不这样用，存疑。
        # sess.run(optimizer, feed_dict={X: train_X, Y:train_Y})


        # Display logs per epoch step
        # 每display_step次输出状态
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
