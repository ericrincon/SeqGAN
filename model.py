import tensorflow as tf


class Discriminator:
    """
    A deep discriminator model for classfiying sequences as real or not as described in the paper
    SeqGAN
    """
    def __init__(self, vocab_size, embedding_size, max_seq_length, filter_windows, nb_filters, nb_classes):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq_length = max_seq_length
        self.filter_windows = filter_windows
        self.nb_filters = nb_filters
        self.nb_classes = nb_classes


    def _build_embedding_layer(self, layer_input, vocab_size, embedding_size, layer_name='embedding_layer'):
        with tf.device('/cpu:0'), tf.name_scope(layer_name):
            W_embeddings = tf.Variable(
                initial_value=tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name='W_embedding'
            )

            embeddings = tf.nn.embedding_lookup(W_embeddings, layer_input)

        return embeddings
    def _build_convolution_layer(self, layer_input, max_sequence_length, filter_windows, nb_filters, embedding_size,
                                 layer_name='Convolution'):

        with tf.device('/gpu:0'), tf.name_scope(layer_name):
            pooled_features = []

            for filter_window_size in filter_windows:
                W_filters = tf.Variable(tf.truncated_normal([None, filter_window_size, embedding_size], stddev=1.0),
                                        name='W_filters' + '_' + str(filter_window_size))
                W_convolution = tf.nn.conv1d(
                    value=layer_input,
                    filters=W_filters,
                    stride=1,
                    padding='VALID',
                    name='W_convolution' + '_' + str(filter_window_size)
                )

                activation = tf.nn.relu(W_convolution, name='Convolution activation' + '_' + str(filter_window_size))

                max_pooling = tf.nn.max_pool(
                    value=activation,
                    ksize=[1, max_sequence_length - filter_window_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='W_max_pooling' + '_' + str(filter_window_size)
                )

                pooled_features.append(max_pooling)

            concatenated_features = tf.concat(3, pooled_features)

        return tf.reshape(concatenated_features, [-1, nb_filters * len(pooled_features)])


    def _build_softmax_layer(self, layer_input, nb_classes):
        W_softmax = tf.Variable(tf.truncated_normal([layer_input.shape[0], nb_classes]))
        b_softmax = tf.Variable(tf.constant(1, shape=nb_classes))

        logits = tf.matmul(layer_input, W_softmax) + b_softmax

        return tf.nn.softmax_cross_entropy_with_logits(logits, )


    def build_model(self):
        X_input = tf.placeholder(tf.float32, shape=[], name='X_input')
        embedding_layer = self._build_embedding_layer(X_input, self.vocab_size, self.embedding_size)
        convolution_layer = self._build_convolution_layer(embedding_layer, self.max_seq_length, self.filter_windows,
                                                    self.nb_filters, self.embedding_size)
        softmax_layer = self._build_softmax_layer(convolution_layer)
class Generator:
