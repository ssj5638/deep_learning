# 선형회귀 : Linear Regression
# 실제 weight에 따른 cost값을 찾는 것

import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
# placeholder의 전달 파라미터
# placeholder(              다른 텐서를 할당
#     dtype=tf.float32,     데이터 타입을 의미하며 반드시 적어야 함
#     shape=None,           입력 데이터의 형태를 의미, 상수값/다차원 배열의 정보가 가능 (default=None)
#     name=None             해당 placeholder의 이름을 부여 (default=None)
# )

hypothesis = X + W

# cost / loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Launch the graph in a session
sess = tf.Session()

# Variables for plotting cost function
W_history = []
cost_history = []

for i in range(-30, 50):
    curr_W = i + 0.1
    curr_cost = sess.run(cost, feed_dict={W:curr_W})
    W_history.append(curr_W)
    cost_history.append(curr_cost)

# Show the cost function
plt.plot(W_history, cost_history)
plt.show()
