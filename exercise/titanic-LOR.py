#%% 
import pandas as pd
import numpy as np
from pandas import DataFrame

data = pd.read_csv("D:\\KAGGLE\\Titanic\\train.csv")

def MaxMinNormalization(x):
    Max = np.max(x)
    Min = np.min(x)
    result = (x - Min) / (Max - Min)
    return result

# 基本数据处理
data['Sex'] = data['Sex'].apply(lambda s:1 if s == 'male' else 0)
data['Age'] = data.Age.fillna(data.Age.mean())
data['Age'] = MaxMinNormalization(data['Age'])
data['Fare'] = MaxMinNormalization(data['Fare'])

data['Dead'] = data['Survived'].apply(lambda s:int(not s))
data = data.join(pd.get_dummies(data.Embarked, prefix='Embarked'))
# data.drop(['Embarked'], axis=1, inplace=True)
data.info()

data_x = data.filter(regex='Pclass|Sex|Fare|Embarked_.*|Age|SibSp|Parch')
# data_x = data[['Pclass', 'Sex', 'Fare', 'Embarked', 'Age', 'SibSp', 'Parch']]
data_y = data[['Dead', 'Survived']]

from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(data_x, data_y, test_size=0.2, random_state=1)
train_X.head()

#%%
import tensorflow as tf

learning_rate = 0.003
training_epochs = 1000
display_step = (int)(training_epochs / 10)
batch_size = 50
data_width = train_X.shape[1]
data_len = train_X.shape[0]

x = tf.placeholder(tf.float32, [None, data_width])
y = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.zeros([data_width, 2]))
b = tf.Variable(tf.zeros([2]))

# 预测函数
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
# loss函数
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

saver = tf.train.Saver()
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(data_len/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            if i == total_batch-1:
                batch_xs = train_X[i*batch_size:data_len]
                batch_ys = train_Y[i*batch_size:data_len]
            else:
                batch_xs = train_X[i*batch_size:(i+1)*batch_size]
                batch_ys = train_Y[i*batch_size:(i+1)*batch_size]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss 平均cost
            avg_cost += c / total_batch

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: test_X, y: test_Y}))

    # save model
    saver.save(sess, "D:\KAGGLE\Titanic\save\model.ckpt")

#%%

data = pd.read_csv("D:\\KAGGLE\\Titanic\\test.csv")

data['Sex'] = data['Sex'].apply(lambda s:1 if s == 'male' else 0)
data['Age'] = data.Age.fillna(data.Age.mean())
data['Age'] = MaxMinNormalization(data['Age'])
data['Fare'] = MaxMinNormalization(data['Fare'])

data = data.join(pd.get_dummies(data.Embarked, prefix='Embarked'))

data_x = data.filter(regex='Pclass|Sex|Fare|Embarked_.*|Age|SibSp|Parch')

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # restore model
    saver.restore(sess, "D:\KAGGLE\Titanic\save\model.ckpt")

    # Test model
    prediction = sess.run(pred, feed_dict={x:data_x})
    Survived = np.argmax(prediction, 1)
    submission = pd.DataFrame({
        "PassengerId": data["PassengerId"],
        "Survived": Survived
    })

    submission.to_csv("D:\\KAGGLE\\Titanic\\titanic_submission.csv", index=False)