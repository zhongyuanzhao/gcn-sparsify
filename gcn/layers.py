from .inits import *

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.compat.v1.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.compat.v1.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.compat.v1.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def maxpooling(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    featuresize = y.shape[1]
    res_cols = []
    if sparse:
        for i in range(featuresize):
            diag = tf.compat.v1.diag(y[:,i])
            imtx = tf.compat.v1.sparse_tensor_dense_matmul(x, diag)
            icol = tf.reduce_max(imtx, axis=1)
            res_cols.append(icol)
        res = tf.reshape(tf.concat(res_cols, axis=0),(-1,featuresize))
    else:
        for i in range(featuresize):
            diag = tf.compat.v1.diag(y[:,i])
            imtx = tf.matmul(x, diag)
            icol = tf.reduce_max(imtx, axis=1)
            res_cols.append(icol)
        res = tf.reshape(tf.concat(res_cols, axis=0),(-1,featuresize))
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, rate=self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class MaxGate(Layer):
    """Dense layer."""
    def __init__(self, input_dim, rank, sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(MaxGate, self).__init__(**kwargs)

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.rank = rank

        # helper variable for sparse dropout
        # self.vars['ones'] = tf.constant(tf.ones([input_dim, input_dim], dtype=tf.float32),
        #                                    name='ones')

        with tf.variable_scope(self.name + '_vars'):
            if self.bias:
                self.vars['bias'] = zeros([input_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        output = inputs
        # transform
        # output = dot(x, self.vars['ones'], sparse=self.sparse_inputs)
        if self.sparse_inputs:
            output = tf.sparse.to_dense(output)
        select = tf.argmax(output, axis=-1)
        output = tf.gather(output, select, batch_dims=self.rank-2)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class KPoly(Layer):
    """Dense layer."""
    def __init__(self, input_dim, rank, sparse_inputs=False,
                 featureless=False, **kwargs):
        super(KPoly, self).__init__(**kwargs)

        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.input_dim = input_dim
        self.rank = rank

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        if self.sparse_inputs:
            sparse_input = tf.cast(inputs, dtype=tf.float32)
            dense_input = tf.compat.v1.sparse_tensor_dense_matmul(sparse_input, tf.eye(self.input_dim))
        else:
            dense_input = inputs
        in_list = []
        for j in range(self.input_dim):
            for i in range(self.rank):
                rank_i = dense_input[:, j:j+1] ** i
                in_list.append(rank_i)

        output = tf.concat(in_list, axis=-1)

        return output


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., channel=0, num_channels=1,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.channel = int(channel)
        self.num_channels = int(num_channels)
        self.order = int(len(placeholders['support']) / self.num_channels)
        self.support = placeholders['support'][self.channel * self.order: self.channel*self.order+self.order]

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                if FLAGS.wts_init == 'random':
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                            name='weights_' + str(i))
                elif FLAGS.wts_init == 'zeros':
                    self.vars['weights_' + str(i)] = zeros([input_dim, output_dim],
                                                            name='weights_' + str(i))
                else:
                    raise NameError('Unsupported wts_init: {}'.format(FLAGS.wts_init))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, rate=self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        # concated = tf.concat([output, self.act(output)], axis=1)
        # return tf.layers.dense(concated, output.shape[1])
        return self.act(output)


class MVGraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, channel_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, aggregator='sum',
                 featureless=False, transpose=False, **kwargs):
        super(MVGraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.transpose = transpose
        self.bias = bias
        self.output_dim = output_dim
        self.channel_dim = int(channel_dim)
        self.order = int(len(placeholders['support'])/channel_dim)
        self.support = placeholders['support']
        self.support_cc = placeholders['support_cc']
        self.aggregator = aggregator.lower()
        assert(len(self.support) % channel_dim == 0)
        assert(sparse_inputs is False)

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        self.mlp_layers = []
        if self.aggregator == 'dense':
            self.mlp_layers.append(Dense(input_dim=channel_dim,
                                         output_dim=1,
                                         placeholders=placeholders,
                                         act=act,
                                         dropout=True,
                                         sparse_inputs=False,
                                         logging=self.logging))

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                if FLAGS.wts_init == 'random':
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                            name='weights_' + str(i))
                elif FLAGS.wts_init == 'zeros':
                    self.vars['weights_' + str(i)] = zeros([input_dim, output_dim],
                                                            name='weights_' + str(i))
                else:
                    raise NameError('Unsupported wts_init: {}'.format(FLAGS.wts_init))
            for i in range(len(self.support_cc)):
                sid = len(self.support) + i
                if FLAGS.wts_init == 'random':
                    self.vars['weights_' + str(sid)] = glorot([output_dim, output_dim],
                                                            name='weights_' + str(sid))
                elif FLAGS.wts_init == 'zeros':
                    self.vars['weights_' + str(sid)] = zeros([output_dim, output_dim],
                                                            name='weights_' + str(sid))
                else:
                    raise NameError('Unsupported wts_init: {}'.format(FLAGS.wts_init))
            if self.bias:
                if self.aggregator == 'concat' or self.aggregator == 'clique':
                    self.vars['bias'] = zeros([output_dim, channel_dim], name='bias')
                else:
                    self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, rate=self.dropout)

        supports = list()
        for i in range(len(self.support)):
            # x_ch = x_channels[int(i/self.order)]
            ic = int(i / self.order)
            x_ch = x[:, :, ic] # channel is the last dimension
            if not self.featureless:
                pre_sup = dot(x_ch, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        ch_outputs = list()
        for i in range(self.channel_dim):
            bidx = self.order * i
            ch_output = supports[bidx:bidx+self.order]
            ch_outputs.append(tf.add_n(ch_output))
        c_out = tf.concat(ch_outputs, axis=-1)
        c_out = tf.reshape(c_out, [-1, self.channel_dim, self.output_dim])
        # c_out = tf.transpose(c_out, [0, 2, 1])

        if self.aggregator == 'max':
            c_agg = tf.reduce_max(c_out, axis=1)
        elif self.aggregator == 'mean':
            c_agg = tf.reduce_mean(c_out, axis=1)
        elif self.aggregator == 'sum':
            c_agg = tf.reduce_sum(c_out, axis=1)
        elif self.aggregator == 'dense':
            c_agg = tf.transpose(c_out, [0, 2, 1])
            for l in self.mlp_layers:
                c_agg = l(c_agg)
            c_agg = tf.reshape(c_agg, [-1, self.output_dim])
        elif self.aggregator == 'clique':
            sid = len(self.support)
            supports_cc = []
            for i in range(len(self.support_cc)):
                pre_sup = dot(c_out, self.vars['weights_' + str(i+sid)], sparse=False)
                support = tf.matmul(self.support_cc[i], pre_sup)
                supports_cc.append(support)
            c_agg = tf.add_n(supports_cc)
            c_agg = tf.transpose(c_agg, [0, 2, 1])
        else:
            c_agg = tf.transpose(c_out, [0, 2, 1])
        output = c_agg

        # bias
        if self.bias:
            output += self.vars['bias']

        if self.transpose:
            output = tf.transpose(output, [0, 2, 1])
        # concated = tf.concat([output, self.act(output)], axis=1)
        # return tf.layers.dense(concated, output.shape[1])
        return self.act(output)


class MaxPoolingAggregator(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, concat=True,
                 featureless=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.adj = placeholders['adj']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.concat = concat

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=input_dim,
                                     output_dim=input_dim,
                                     placeholders=placeholders,
                                     act=act,
                                     dropout=True,
                                     sparse_inputs=self.sparse_inputs,
                                     logging=self.logging))

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            for i in range(2):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        # self
        if not self.featureless:
            pre_sup = dot(x, self.vars['weights_' + str(0)],
                          sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_' + str(0)]
        support = dot(self.support[0], pre_sup, sparse=True)
        supports.append(support)
        # neighbors
        if not self.featureless:
            pre_sup = x
        else:
            pre_sup = tf.ones_like(x, dtype=tf.float32)
        for l in self.mlp_layers:
            pre_sup = l(pre_sup)
        neigh_h = maxpooling(self.adj, pre_sup, sparse=True)
        support = dot(neigh_h, self.vars['weights_' + str(1)],
                      sparse=False)
        supports.append(support)

        if not self.concat:
            output = tf.add_n(supports)
        else:
            output = tf.concat(supports, axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


# class MaxPoolingAggregator(Layer):
#     """ Aggregates via max-pooling over MLP functions.
#     """
#
#     def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
#                  dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
#         super(MaxPoolingAggregator, self).__init__(**kwargs)
#
#         self.dropout = dropout
#         self.bias = bias
#         self.act = act
#         self.concat = concat
#
#         if neigh_input_dim is None:
#             neigh_input_dim = input_dim
#
#         if name is not None:
#             name = '/' + name
#         else:
#             name = ''
#
#         if model_size == "small":
#             hidden_dim = self.hidden_dim = 512
#         elif model_size == "big":
#             hidden_dim = self.hidden_dim = 1024
#
#         self.mlp_layers = []
#         self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
#                                      output_dim=hidden_dim,
#                                      act=tf.nn.relu,
#                                      dropout=dropout,
#                                      sparse_inputs=False,
#                                      logging=self.logging))
#
#         with tf.variable_scope(self.name + name + '_vars'):
#             self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
#                                                 name='neigh_weights')
#
#             self.vars['self_weights'] = glorot([input_dim, output_dim],
#                                                name='self_weights')
#             if self.bias:
#                 self.vars['bias'] = zeros([self.output_dim], name='bias')
#
#         if self.logging:
#             self._log_vars()
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.neigh_input_dim = neigh_input_dim
#
#     def _call(self, inputs):
#         self_vecs, neigh_vecs = inputs
#         neigh_h = neigh_vecs
#
#         dims = tf.shape(neigh_h)
#         batch_size = dims[0]
#         num_neighbors = dims[1]
#         # [nodes * sampled neighbors] x [hidden_dim]
#         h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))
#
#         for l in self.mlp_layers:
#             h_reshaped = l(h_reshaped)
#         neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
#         neigh_h = tf.reduce_max(neigh_h, axis=1)
#
#         from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
#         from_self = tf.matmul(self_vecs, self.vars["self_weights"])
#
#         if not self.concat:
#             output = tf.add_n([from_self, from_neighs])
#         else:
#             output = tf.concat([from_self, from_neighs], axis=1)
#
#         # bias
#         if self.bias:
#             output += self.vars['bias']
#
#         return self.act(output)


class NoisyDense(Layer):
    """
    Factorized Gaussian Noise Layer
    Reference from https://github.com/Kaixhin/Rainbow/blob/master/model.py
    """
    def __init__(self, units, input_dim, std_init=0.5, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.std_init = std_init
        self.reset_noise(input_dim)
        mu_range = 1 / np.sqrt(input_dim)
        sigma_val = self.std_init / np.sqrt(self.units)
        # mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        # sigma_initializer = tf.constant_initializer(self.std_init / np.sqrt(self.units))
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weight_mu'] = uniform((input_dim, units), mu_range, 'weight_mu')
            self.vars['weight_sigma'] = sigma_val * ones((input_dim, units), 'weight_sigma')
            self.vars['bias_mu'] = uniform((units,), mu_range, 'bias_mu')
            self.vars['bias_sigma'] = sigma_val * ones((units,), 'bias_sigma')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # output = tf.tensordot(inputs, self.kernel, 1)
        # tf.nn.bias_add(output, self.bias)
        # return output
        self.kernel = self.vars['weight_mu'] + self.vars['weight_sigma'] * self.weights_eps
        self.bias = self.vars['bias_mu'] + self.vars['bias_sigma'] * self.bias_eps
        return tf.matmul(inputs, self.kernel) + self.bias

    def _scale_noise(self, dim):
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    def reset_noise(self, input_shape):
        eps_in = self._scale_noise(input_shape)
        eps_out = self._scale_noise(self.units)
        self.weights_eps = tf.multiply(tf.expand_dims(eps_in, 1), eps_out)
        self.bias_eps = eps_out

