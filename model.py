import tensorflow as tf
import numpy as np
import sys


class Generator:
    """
    A simple LSTM network for sequence generation.
    """
    def __init__(self, nb_units, vocab_size, embedding_size, max_seq_length):
        self.nb_units = nb_units
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq_length = max_seq_length

        self.X_input = tf.placeholder(tf.int32, [None, max_seq_length])
        self.y_output = tf.placeholder(tf.int32, [None, max_seq_length])


        with tf.device('/cpu:0'), tf.name_scope('embeddings'):
            W_embeddings = tf.Variable(
                initial_value=tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name='W_embedding'
            )

            self.embeddings = tf.nn.embedding_lookup(W_embeddings, self.X_input)

        rnn = tf.contrib.rnn.LSTMBlockCell(
            num_units=nb_units,
            forget_bias=1.0,
            use_peephole=False
        )

        self.lstm = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([rnn] * 1)

        self.global_step = tf.Variable(0, trainable=False)


    def train(self, X, y, nb_epochs, batch_size=32, learning_rate=.001):
        self.initial_state = self.state = tf.zeros([batch_size, self.nb_units])
        self.W_softmax = tf.get_variable('W_softmax', [batch_size, self.nb_units, self.vocab_size])
        self.b_softmax = tf.get_variable('b_softmax', [self.vocab_size])

        with tf.variable_scope('RNN') as scope:
            output, hidden_state = tf.nn.dynamic_rnn(self.lstm, self.embeddings, dtype=tf.float32)

        logits = tf.matmul(output, self.W_softmax) + self.b_softmax

        # logits: A 3D Tensor of shape [batch_size x sequence_length x num_decoder_symbols] and dtype float.
        # The logits correspond to the prediction across all classes at each timestep.
        # targets: A 2D Tensor of shape [batch_size x sequence_length] and dtype int. The target represents
        # the true class at each timestep.
        weights = tf.ones([batch_size, self.max_seq_length])

        loss_per_seq = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=self.y_output, weights=weights)
        loss = tf.reduce_mean(loss_per_seq)
        
        with tf.variable_scope('loss') as scope:
            scope.reuse_variables()
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)

        # Gradient clipping
        grads, variables = zip(*grads_and_vars)
        grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=5.0)
        grads_and_vars = zip(grads_clipped, variables)

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        tf_vars_to_save = tf.trainable_variables() + [self.global_step]
        saver = tf.train.Saver(tf_vars_to_save, max_to_keep=5)

        init = tf.initialize_all_variables()

        total_loss = float(0)

        with tf.Session() as sess:
            sess.run(init)

            epoch_i = 0

            while epoch_i < nb_epochs:
                batch_i = 0
                batch_losses = []
                numpy_state = self.initial_state.eval()

                for i in range(batch_size, X.shape[0], batch_size):
                    X_batch, y_batch = X[batch_i:i], y[batch_i:i]

                    sess.run(train_op, feed_dict={
                        self.X_input: X_batch,
                        self.y_output: y_batch
                    })

                    cost = sess.run([loss], feed_dict={
                        self.X_input: X_batch,
                        self.y_output: y_batch
                    })

                    total_loss += cost[0]
                    batch_losses.append(total_loss)

                    batch_i = i
                print('Epoch: {} loss: {}'.format(epoch_i + 1, mean(batch_losses)))

                epoch_i += 1





class Discriminator:
    """
    A deep discriminator model for classifying sequences as real or not as described in the paper
    SeqGAN
    """
    def __init__(self, vocab_size, embedding_size, max_seq_length, filter_windows, nb_filters, nb_classes):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq_length = max_seq_length
        self.filter_windows = filter_windows
        self.nb_filters = nb_filters
        self.nb_classes = nb_classes

        self.X_input = tf.placeholder(tf.int32, shape=[None, self.max_seq_length], name='X_input')
        self.y_input = tf.placeholder(tf.float32, shape=[None, 2], name='y_input')

        self.build_model()

    def _build_embedding_layer(self, layer_input, vocab_size, embedding_size, layer_name='embedding_layer'):
        with tf.device('/cpu:0'), tf.name_scope(layer_name):
            W_embeddings = tf.Variable(
                initial_value=tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name='W_embedding'
            )

            self.embeddings = tf.nn.embedding_lookup(W_embeddings, layer_input)

        return tf.expand_dims(self.embeddings, -1)

    def _build_convolution_layer(self, layer_input, max_sequence_length, filter_windows, nb_filters, embedding_size,
                                 layer_name='Convolution'):

        pooled_features = []

        for filter_window_size in filter_windows:
            W_filters = tf.Variable(
                tf.truncated_normal([filter_window_size, embedding_size, 1, nb_filters], stddev=1.0),
                                    name='W_filters' + '_' + str(filter_window_size)
            )
            b = tf.Variable(tf.constant(.1, shape=[nb_filters]))

            W_convolution = tf.nn.conv2d(
                input=layer_input,
                filter=W_filters,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='W_convolution' + '_' + str(filter_window_size)
            )

            activation = tf.nn.relu(tf.nn.bias_add(W_convolution, b), name='Convolution_activation' + '_' + str(filter_window_size))

            max_pooling = tf.nn.max_pool(
                value=activation,
                ksize=[1, max_sequence_length - filter_window_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='W_max_pooling' + '_' + str(filter_window_size)
            )

            pooled_features.append(max_pooling)

        concatenated_features = tf.concat(3, pooled_features)
        self.final_feature_length = nb_filters * len(pooled_features)

        return tf.reshape(concatenated_features, [-1, self.final_feature_length])


    def _build_softmax_layer(self, layer_input, nb_classes):
        with tf.name_scope('softmax_output'):
            W_softmax = tf.Variable(tf.truncated_normal([self.final_feature_length, nb_classes]))
            b_softmax = tf.Variable(tf.constant(0.1, shape=[nb_classes]))

            scores = tf.matmul(layer_input, W_softmax) + b_softmax
            predictions = tf.argmax(scores, 1)

        return scores, predictions



    def build_model(self):
        self.embedding_layer = self._build_embedding_layer(self.X_input, self.vocab_size, self.embedding_size)
        self.convolution_layer = self._build_convolution_layer(self.embedding_layer, self.max_seq_length,
                                                               self.filter_windows, self.nb_filters,
                                                               self.embedding_size)
        self.scores, self.predictions = self._build_softmax_layer(self.convolution_layer, self.nb_classes)




    def train(self, X, y, nb_epochs, batch_size=32, learning_rate=.001):


        # Evaluate model
        correct_pred = tf.equal(self.predictions, tf.argmax(self.y_input, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y_input))

        with tf.name_scope('loss'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            epoch_i = 0

            while epoch_i < nb_epochs:
                batch_i = 0
                batch_losses = []
                batch_accs = []

                for i in range(batch_size, X.shape[0], batch_size):
                    X_batch, y_batch = X[batch_i:i], y[batch_i:i]

                    sess.run(optimizer, feed_dict={
                        self.X_input: X_batch,
                        self.y_input: y_batch
                    })

                    loss, acc = sess.run([cost, accuracy], feed_dict={
                        self.X_input: X_batch,
                        self.y_input: y_batch
                    })

                    batch_accs.append(acc)
                    batch_losses.append(loss)

                    batch_i = i
                print('Epoch: {} loss: {:.6f} acc: {:.6f}'.format(epoch_i + 1, mean(batch_losses), mean(batch_accs)))

                epoch_i += 1

def mean(X):
    average = float(0)

    for x in X:
        average += x

    average /= len(X)

    return average