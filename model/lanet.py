import tensorflow as tf
import numpy as np

from sklearn.utils import shuffle


class LaNet:

    def __init__(self, n_out, mu=0, sigma=0.1, learning_rate=0.001):
        # -------------------------------------------------------------------------------------------------
        # Hyperparameters
        self.mu = mu
        self.sigma = sigma
        self.n_out = n_out
        self.learning_rate = learning_rate
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, 32, 32, 1], name='x')
        self.y = tf.compat.v1.placeholder(tf.int32, shape=[None], name='y')
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        self.keep_prob_conv = tf.compat.v1.placeholder(tf.float32, name='keep_prob_conv')

        # -------------------------------------------------------------------------------------------------
        # Layer 1 (CNN): Input = 32x32x1 -> Filter = 5x5x6 -> Output = 28x28x6
        # Weight and bias
        self.conv1_weight = tf.Variable(tf.random.truncated_normal(shape=(5, 5, 1, 6),
                                                                   mean=self.mu,
                                                                   stddev=self.sigma))
        self.conv1_bias = tf.Variable(tf.zeros(6))
        # Apply CNN
        self.conv1 = tf.nn.conv2d(self.x,
                                  self.conv1_weight,
                                  strides=[1, 1, 1, 1],
                                  padding='VALID') + self.conv1_bias
        # Activation
        self.conv1 = tf.nn.relu(self.conv1)
        # Max pooling layer: Input = 28x28x6 -> Output 14x14x6
        self.conv1 = tf.nn.max_pool2d(self.conv1,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='VALID')

        # -------------------------------------------------------------------------------------------------
        # Layer 2 (CNN): Input 14x14x6 -> Filter = 5x5x16 -> Output = 10x10x16
        self.conv2_weight = tf.Variable(tf.random.truncated_normal(shape=(5, 5, 6, 16),
                                                                   mean=self.mu,
                                                                   stddev=self.sigma))
        self.conv2_bias = tf.Variable(tf.zeros(16))
        # Apply CNN
        self.conv2 = tf.nn.conv2d(self.conv1,
                                  self.conv2_weight,
                                  strides=[1, 1, 1, 1],
                                  padding='VALID') + self.conv2_bias
        # Activation
        self.conv2 = tf.nn.relu(self.conv2)
        # Max pooling layer: Input = 10x10x16 -> Output = 5x5x16
        self.conv2 = tf.nn.max_pool2d(self.conv2,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='VALID')

        # -------------------------------------------------------------------------------------------------
        # Flattening: Input = 5x5x16 -> Output = 400
        self.fully_connected1 = tf.compat.v1.keras.layers.Flatten()(self.conv2)

        # -------------------------------------------------------------------------------------------------
        # Layer 3 (Fully Connected): Input = 400 -> Output = 120
        self.fully_connected2_weight = tf.Variable(tf.random.truncated_normal(shape=(400, 120),
                                                                              mean=self.mu,
                                                                              stddev=self.sigma))
        self.fully_connected2_bias = tf.Variable(tf.zeros(120))
        # Apply Fully Connected
        self.fully_connected2 = tf.add((tf.matmul(self.fully_connected1, self.fully_connected2_weight)),
                                       self.fully_connected2_bias)
        # Activation
        self.fully_connected2 = tf.nn.relu(self.fully_connected2)

        # -------------------------------------------------------------------------------------------------
        # Layer 4 (Fully Connected): Input = 120 -> Output = 84
        self.fully_connected3_weight = tf.Variable(tf.random.truncated_normal(shape=(120, 84),
                                                                              mean=self.mu,
                                                                              stddev=self.sigma))
        self.fully_connected3_bias = tf.Variable(tf.zeros(84))
        # Apply Fully Connected
        self.fully_connected3 = tf.add((tf.matmul(self.fully_connected2, self.fully_connected3_weight)),
                                       self.fully_connected3_bias)
        # Activation
        self.fully_connected3 = tf.nn.relu(self.fully_connected3)

        # -------------------------------------------------------------------------------------------------
        # Layer 5 (Fully Connected): Input = 84 -> Output = n_out (76 in this case)
        self.output_weights = tf.Variable(tf.random.truncated_normal(shape=(84, self.n_out),
                                                                     mean=self.mu,
                                                                     stddev=self.sigma))
        self.output_bias = tf.Variable(tf.zeros(self.n_out))
        # Apply Fully Connected
        self.logits = tf.add((tf.matmul(self.fully_connected3, self.output_weights)),
                             self.output_bias)

        # Training operation
        self.one_hot_y = tf.one_hot(self.y, self.n_out)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                        labels=self.one_hot_y)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        self.correct_prediction = tf.equal(tf.math.argmax(self.logits, 1),
                                           tf.math.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.saver = tf.train.Saver()

    def predict(self, x_test, batch_size):
        num_examples = len(x_test)
        y_pred = np.zeros(num_examples, dtype=np.int32)
        sess = tf.compat.v1.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x = x_test[offset:offset + batch_size]
            y_pred[offset:offset + batch_size] = sess.run(tf.math.argmax(self.logits, 1),
                                                          feed_dict={self.x: batch_x,
                                                                     self.keep_prob: 1,
                                                                     self.keep_prob_conv: 1})
        return y_pred

    def evaluate(self, x_test, y_test, batch_size):
        num_examples = len(x_test)
        total_accuracy = 0
        sess = tf.compat.v1.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = x_test[offset:offset + batch_size], y_test[offset:offset + batch_size]
            accuracy = sess.run(self.accuracy_operation,
                                feed_dict={self.x: batch_x,
                                           self.y: batch_y,
                                           self.keep_prob: 1,
                                           self.keep_prob_conv: 1})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def training(self, x_train, y_train, x_valid, y_valid, epochs, batch_size):
        with tf.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            num_examples = len(x_train)
            print("TRAINING...")
            for i in range(epochs):
                x_train, y_train = shuffle(x_train, y_train)
                for offset in range(0, num_examples, batch_size):
                    batch_x, batch_y = x_train[offset:offset + batch_size], y_train[offset:offset + batch_size]
                    sess.run(self.training_operation, feed_dict={self.x: batch_x,
                                                                 self.y: batch_y,
                                                                 self.keep_prob: 0.5,
                                                                 self.keep_prob_conv: 0.7})

                valid_accuracy = self.evaluate(x_valid, y_valid, batch_size)
                print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i + 1, (valid_accuracy * 100)))
            self.saver.save(sess, "../logs/LaNet")
            print("MODEL SAVED")
