""" K-Means.

Implement K-Means algorithm with TensorFlow, and apply it to classify
handwritten digit images. This example is using the MNIST database of
handwritten digits as training samples (http://yann.lecun.com/exdb/mnist/).

Note: This example requires TensorFlow v1.1.0 or over.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
#%%
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# Ignore all GPUs, tf k-means does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
full_data_x = mnist.train.images

# Parameters
num_steps = 50 # Total steps to train 循环50次
batch_size = 1024 # The number of samples per batch 无用
k = 25 # The number of clusters 企图分出25个聚类
num_classes = 10 # The 10 digits
num_features = 784 # Each image is 28x28 pixels

# Input images
X = tf.placeholder(tf.float32, shape=[None, num_features])
# Labels (for assigning a label to a centroid and testing)
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# K-Means Parameters
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)
# KMeans class
# __init__(
#     inputs,               输入数据
#     num_clusters,         分类数量
#     initial_clusters=RANDOM_INIT,     初始中心点，可tensor、numpy指定、"random"、"kmeans_plus_plus"
#           random 随机从输入抽取
#     distance_metric=SQUARED_EUCLIDEAN_DISTANCE,   距离计算方式，Supported:"squared_euclidean","cosine"
#     use_mini_batch=False,             对数据分块(batch)迭代，加快收敛，默认不开启，全数据迭代
#     mini_batch_steps_per_iteration=1, （mini—batch模式下）多少次迭代后将子块训练数据更新到主副本（更新中心点）
#     random_seed=0,                    随机化初始中心点的随机种子
#     kmeans_plus_plus_num_retries=2,   （待深入）For each point that is sampled during kmeans++ initialization, this parameter specifies the number of additional points to draw from the current distribution before selecting the best. If a negative value is specified, a heuristic is used to sample O(log(num_to_sample)) additional points.
#     kmc2_chain_length=200             （待深入）Determines how many candidate points are used by the k-MC2 algorithm to produce one new cluster centers. If a (mini-)batch contains less points, one new cluster center is generated from the (mini-)batch.
# )

# Build KMeans graph
# 根据上面创建的KMEANS类的参数，training_graph()构造了k-means算法的graph，
# 并返回各结点(tensor)。无需自己设计计算流程，真方便
training_graph = kmeans.training_graph()
# training_graph():
#   return (all_scores, cluster_idx, scores, cluster_centers_initialized,
#             init_op, training_op)
# all_scores: 每个输入到每个聚类中心的距离矩阵 (len(X), k)
# cluster_idx: 向量，对应每个输入最近的聚类中心id (len(X))，
# scores: 向量，对应每个输入到最近的中心的距离 (len(X))
# cluster_centers_initialized: 标量，中心是否已经初始化
# init_op: 操作，初始化中心
# training_op: 操作，训练

if len(training_graph) > 6: 
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     cluster_centers_var, init_op, train_op) = training_graph
else: #TensorFlow 1.13.1 从上面training_graph()的返回可以看到只有6个变量
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, train_op) = training_graph

cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)
# scores的均值，用以表示整体匹配度

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

# Training
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))

# Assign a label to each centroid
# Count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx')
# counts 通过每个输入样例和样例输出，得到每个聚类中代表10个数字的次数统计 (k, 10)
counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]
# Assign the most frequent label to the centroid
# labels_map 向量，每个聚类出现最多的数字，作为聚类预测结果 (k)
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

# Evaluation ops
# Lookup: centroid_id -> label
# embedding_lookup将cluster_idx每一个元素作为下标在labels_map中查找，获取其值
# 根据labels_map，将cluster_idx从对应每个输入最近中心坐标，变成每个输入预测结果
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# Compute accuracy
# 准确率计算
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
# 计算测试用例，得到准确率
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
