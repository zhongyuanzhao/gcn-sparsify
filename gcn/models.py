from .layers import *
from .metrics import *
from .layers import _LAYER_UIDS
import numpy as np


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'concat'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.skip = FLAGS.skip

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []
        self.utilnets = []
        self.attacts = []
        self.actor_layers = []
        self.critic_layers = []
        self.actors = []
        self.critics = []

        self.inputs = None
        self.outputs = None
        self.outputs_softmax = None
        self.outputs_utility = None
        self.pred = None
        self.output_dim = None
        self.input_dim = None
        self.actor_space = None
        self.critic_space = None

        self.loss = 0
        self.loss_crt = 0
        self.loss_act = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.optimizer = None
        self.opt_op = None
        self.optimizer_crt = None
        self.opt_op_crt = None

    def _build(self):
        raise NotImplementedError

    def _wire(self):
        raise NotImplementedError

    def _opt_set(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        self.sparse_input = tf.cast(self.inputs, dtype=tf.float32)
        self.dense_input = tf.compat.v1.sparse_tensor_dense_matmul(self.sparse_input, tf.eye(self.input_dim))
        self.normed_input = tf.ones_like(self.dense_input)
        self._wire()

        # Store model variables for easy access
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        self._f1()
        self._opt_set()

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _loss_reg(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def _f1(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.compat.v1.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.compat.v1.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class ModelD(object):
    """Model with dense input features"""
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'concat'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.skip = FLAGS.skip

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []
        self.utilnets = []
        self.attacts = []
        self.actor_layers = []
        self.critic_layers = []
        self.actors = []
        self.critics = []

        self.inputs = None
        self.outputs = None
        self.outputs_softmax = None
        self.outputs_utility = None
        self.pred = None
        self.output_dim = None
        self.input_dim = None
        self.actor_space = None
        self.critic_space = None

        self.loss = 0
        self.loss_crt = 0
        self.loss_act = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.optimizer = None
        self.opt_op = None
        self.optimizer_crt = None
        self.opt_op_crt = None

    def _build(self):
        raise NotImplementedError

    def _wire(self):
        raise NotImplementedError

    def _opt_set(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        self.normed_input = tf.ones_like(self.inputs)
        self._wire()

        # Store model variables for easy access
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        self._f1()
        self._opt_set()

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _loss_reg(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def _f1(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.compat.v1.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.compat.v1.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)



class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += my_softmax_cross_entropy(self.outputs, self.placeholders['labels'])
    def _loss_reg(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        self.loss += tf.reduce_mean(tf.square(self.outputs-self.placeholders['labels']))

    def _accuracy(self):
        self.accuracy = my_accuracy(self.outputs, self.placeholders['labels'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class MLP2(Model):
    """Explicitly input hyperparameters rather than from FLAGS """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(MLP2, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step_tensor, 5000,
                                                                      self.learning_decay, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:, 0:self.output_dim])
        diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # diver_loss += self.weight_decay * tf.reduce_mean(self.outputs[:,0:self.output_dim])
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs)
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        # Opt 1: Plain network
        if self.is_dual:
            self.outputs = tf.reduce_mean(self.activations[-1][:, 0], 0) \
                           + (self.activations[-1][:, 1:] - tf.reduce_mean(self.activations[-1][:, 1:], 0))
        else:
            self.outputs = self.activations[-1]
        # Opt 2: Dueling network
        # self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0)
        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if self.num_layer == 1:
            self.layers.append(Dense(input_dim=self.input_dim,
                                     output_dim=self.output_dim,
                                     placeholders=self.placeholders,
                                     act=self.act,
                                     dropout=True,
                                     sparse_inputs=True,
                                     bias=self.bias,
                                     logging=self.logging))
        else:
            self.layers.append(Dense(input_dim=self.input_dim,
                                     output_dim=self.hidden_dim,
                                     placeholders=self.placeholders,
                                     act=self.act,
                                     dropout=True,
                                     sparse_inputs=True,
                                     bias=self.bias,
                                     logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(Dense(input_dim=self.hidden_dim,
                                        output_dim=self.hidden_dim,
                                        placeholders=self.placeholders,
                                        act=self.act,
                                        dropout=True,
                                        bias=self.bias,
                                        logging=self.logging))

            self.layers.append(Dense(input_dim=self.hidden_dim,
                                    output_dim=self.output_dim,
                                    placeholders=self.placeholders,
                                    act=self.act,
                                    # act=lambda x: x,
                                    dropout=True,
                                    bias=self.bias,
                                    logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN_DEEP_DIVER(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_DEEP_DIVER, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        if FLAGS.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
            learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.learning_rate, self.global_step_tensor, 1000, FLAGS.learning_decay, staircase=True)
        else:
            learning_rate = FLAGS.learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        sparse_input = tf.cast(self.inputs, dtype=tf.float32)
        dense_input = tf.compat.v1.sparse_tensor_dense_matmul(sparse_input, tf.eye(self.input_dim))
        # 32 outputs
        # diver_loss = my_softmax_cross_entropy(self.outputs[:,0:self.output_dim], self.placeholders['labels'])
        diver_loss = my_weighted_softmax_cross_entropy(self.outputs[:,0:self.output_dim], self.placeholders['labels'], dense_input[:, 0])
        for i in range(1,FLAGS.diver_num):
            # diver_loss = tf.reduce_min([diver_loss, my_softmax_cross_entropy(self.outputs[:, 2*i:2*i + self.output_dim], self.placeholders['labels'])])
            diver_loss = tf.reduce_min([diver_loss, my_weighted_softmax_cross_entropy(self.outputs[:, 2 * i:2 * i + self.output_dim],
                                                                         self.placeholders['labels'], dense_input[:, 0])])
        self.loss += diver_loss

    def _loss_reg(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        self.loss += tf.reduce_mean(tf.abs(self.outputs-self.placeholders['labels']))

    def _accuracy(self):
        # 32 outputs
        acc = my_accuracy(self.outputs[:,0:self.output_dim], self.placeholders['labels'])
        for i in range(1,FLAGS.diver_num):
            acc = tf.reduce_max([acc, my_accuracy(self.outputs[:,2*i:2*i+self.output_dim], self.placeholders['labels'])])
        self.accuracy = acc

    def _f1(self):
        # 32 outputs
        f1, precision, recall = my_f1(self.outputs[:,0:self.output_dim], self.placeholders['labels'])
        for i in range(1,FLAGS.diver_num):
            f1_i, prec_i, recall_i = my_f1(self.outputs[:,2*i:2*i+self.output_dim], self.placeholders['labels'])
            f1 = tf.reduce_max([f1, f1_i])
            precision = tf.reduce_max([precision, prec_i])
            recall = tf.reduce_max([recall, recall_i])
        self.f1 = f1
        self.precision = precision
        self.recall = recall

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        if not self.skip:
            self.outputs = self.activations[-1]
        else:
            # hiddens = [dense_input] + self.activations[1:]
            hiddens = [self.dense_input] + self.activations[-1:]
            super_hidden = tf.concat(hiddens, axis=1)
            if FLAGS.wts_init == 'random':
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1])
            elif FLAGS.wts_init == 'zeros':
                input_dim = self.dense_input.get_shape().as_list()[1]
                output_dim = self.activations[-1].shape.as_list()[1]
                dense_shape = [super_hidden.get_shape().as_list()[1], output_dim]
                init_wts = np.zeros(dense_shape, dtype=np.float32)
                diag_mtx = np.identity(int(output_dim/2))
                neg_indices = list(range(0, output_dim-1, 2))
                pos_indices = list(range(1, output_dim, 2))
                init_wts[0:int(output_dim/2), neg_indices] = - diag_mtx
                init_wts[0:int(output_dim/2), pos_indices] = diag_mtx
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1], kernel_initializer=tf.constant_initializer(init_wts))

        self.outputs_softmax = tf.nn.softmax(self.outputs[:,0:2])
        for out_id in range(1, FLAGS.diver_num):
            self.outputs_softmax = tf.concat([self.outputs_softmax, tf.nn.softmax(self.outputs[:,self.output_dim*out_id:self.output_dim*(out_id+1)])], axis=1)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        for i in range(FLAGS.num_layer-2):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.leaky_relu,
                                                dropout=True,
                                                logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=2*FLAGS.diver_num,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs)
        return self.outputs_softmax


class GCN_DQN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        if FLAGS.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                      initializer=tf.zeros_initializer)
            learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.learning_rate, self.global_step_tensor, 5000,
                                                       FLAGS.learning_decay, staircase=True)
        else:
            learning_rate = FLAGS.learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        # mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:,0:self.output_dim])
        # diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs[:,0:self.output_dim])

        for i in range(1, FLAGS.diver_num):
            diver_loss = tf.reduce_min([diver_loss,
                                        tf.reduce_mean(
                                            tf.abs(self.outputs[:, i:i + self.output_dim] - self.placeholders['labels']))])
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        if not self.skip:
            self.outputs = self.activations[-1]
        else:
            # hiddens = [dense_input] + self.activations[1:]
            hiddens = [self.dense_input] + self.activations[-1:]
            super_hidden = tf.concat(hiddens, axis=1)
            if FLAGS.wts_init == 'random':
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1])
            elif FLAGS.wts_init == 'zeros':
                input_dim = self.dense_input.get_shape().as_list()[1]
                output_dim = self.activations[-1].shape.as_list()[1]
                dense_shape = [super_hidden.get_shape().as_list()[1], output_dim]
                init_wts = np.zeros(dense_shape, dtype=np.float32)
                diag_mtx = np.identity(int(output_dim/2))
                neg_indices = list(range(0, output_dim-1, 2))
                pos_indices = list(range(1, output_dim, 2))
                init_wts[0:int(output_dim/2), neg_indices] = - diag_mtx
                init_wts[0:int(output_dim/2), pos_indices] = diag_mtx
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1], kernel_initializer=tf.constant_initializer(init_wts))

        # self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0)
        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if FLAGS.num_layer==1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.diver_num,
                                                placeholders=self.placeholders,
                                                # act=tf.nn.leaky_relu,
                                                act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                # bias=True,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.leaky_relu,
                                                # act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                logging=self.logging))
            for i in range(FLAGS.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                    output_dim=FLAGS.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.leaky_relu,
                                                    # act=lambda x: x,
                                                    dropout=True,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.diver_num,
                                                placeholders=self.placeholders,
                                                # act=tf.nn.leaky_relu,
                                                act=lambda x: x,
                                                dropout=True,
                                                logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN2_DQN(Model):
    """Explicitly input hyperparameters rather than from FLAGS """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(GCN2_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step_tensor, 5000,
                                                                      self.learning_decay, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:, 0:self.output_dim])
        diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # diver_loss += self.weight_decay * tf.reduce_mean(self.outputs[:,0:self.output_dim])
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs)
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        # Opt 1: Plain network
        if self.is_dual:
            self.outputs = tf.reduce_mean(self.activations[-1][:, 0], 0) \
                           + (self.activations[-1][:, 1:] - tf.reduce_mean(self.activations[-1][:, 1:], 0))
        else:
            self.outputs = self.activations[-1]
        # Opt 2: Dueling network
        # self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0)
        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if self.num_layer == 1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                    output_dim=self.hidden_dim,
                                                    placeholders=self.placeholders,
                                                    act=self.act,
                                                    dropout=True,
                                                    bias=self.bias,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                bias=self.bias,
                                                logging=self.logging))

        # self.layers.append(NoisyDense(input_dim=self.hidden_dim,
        #                               units=self.output_dim + 1,
        #                               logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCNe_DQN(Model):
    """Eager execution
    Explicitly input hyperparameters rather than from FLAGS
    """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(GCNe_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual
        self.is_noise = is_noisy

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.Variable(0, trainable=False)
            self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate,
                decay_steps=5000,
                decay_rate=self.learning_decay,
                staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:, 0:self.output_dim])
        diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        # Opt 1: Plain network
        if self.is_dual:
            self.outputs = tf.reduce_mean(self.activations[-1][:, 0], 0) \
                           + (self.activations[-1][:, 1:] - tf.reduce_mean(self.activations[-1][:, 1:], 0))
        else:
            self.outputs = self.activations[-1]
        # Opt 2: Dueling network
        # self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0)
        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if self.num_layer == 1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                    output_dim=self.hidden_dim,
                                                    placeholders=self.placeholders,
                                                    act=self.act,
                                                    dropout=True,
                                                    bias=self.bias,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                bias=self.bias,
                                                logging=self.logging))

        # self.layers.append(NoisyDense(input_dim=self.hidden_dim,
        #                               units=self.output_dim + 1,
        #                               logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN_Fast(Model):
    """Explicitly input hyperparameters rather than from FLAGS """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(GCN_Fast, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step_tensor, 5000,
                                                                      self.learning_decay, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:, 0:self.output_dim])
        diver_loss = tf.reduce_mean(mse, name="loss")
        # diver_loss += self.weight_decay * tf.reduce_mean(self.outputs[:,0:self.output_dim])
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs)
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        const_inputs = tf.ones_like(self.dense_input)
        self.activations.append(const_inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        self.fast_params = self.activations[-1]
        # self.fast_weight = self.fast_params[:, 0:1]
        # self.fast_bias = self.fast_params[:, 1:]
        # self.outputs = self.dense_input * self.fast_weight - self.fast_bias
        self.fast_w0 = self.fast_params[:, 0:8]
        self.fast_b0 = self.fast_params[:, 8:16]
        self.fast_w1 = self.fast_params[:, 16:24]
        self.fast_b1 = self.fast_params[:, 24:25]
        self.fast_l0 = tf.nn.leaky_relu(tf.math.multiply(self.dense_input, self.fast_w0) - self.fast_b0)
        self.fast_l0 = tf.expand_dims(self.fast_l0, axis=1)
        self.fast_w1 = tf.expand_dims(self.fast_w1, axis=2)
        self.outputs = tf.nn.leaky_relu(tf.squeeze(tf.linalg.matmul(self.fast_l0, self.fast_w1), axis=[2])
                                        +self.fast_b1)
        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if self.num_layer == 1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=25, # self.output_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=False,
                                                bias=self.bias,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                dropout=True,
                                                sparse_inputs=False,
                                                bias=self.bias,
                                                logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                    output_dim=self.hidden_dim,
                                                    placeholders=self.placeholders,
                                                    act=self.act,
                                                    dropout=True,
                                                    bias=self.bias,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=25, #self.output_dim,
                                                placeholders=self.placeholders,
                                                # act=self.act,
                                                act=lambda x: x,
                                                dropout=True,
                                                bias=self.bias,
                                                logging=self.logging))

        # self.layers.append(NoisyDense(input_dim=self.hidden_dim,
        #                               units=self.output_dim + 1,
        #                               logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN4_DQN(Model):
    """
    Explicitly input hyperparameters rather than from FLAGS
    Distributed control with binary output at each node
    """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(GCN4_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['actions'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step_tensor, 5000,
                                                                      self.learning_decay, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:,0:self.output_dim])
        diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # diver_loss += self.weight_decay * tf.reduce_mean(self.outputs[:,0:self.output_dim])
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs)
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id == 0:
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            elif layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                hidden = hidden + self.activations[-1]
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        # Opt 1: Plain network
        if self.is_dual:
            self.outputs = tf.reduce_mean(self.activations[-1][:, 0], 0) \
                           + (self.activations[-1][:, 1:] - tf.reduce_mean(self.activations[-1][:, 1:], 0))
        else:
            self.outputs = self.activations[-1]
        # Opt 2: Dueling network
        self.outputs_softmax = tf.nn.softmax(self.outputs, axis=1)
        # self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs_softmax, axis=1)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if self.num_layer == 1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                    output_dim=self.hidden_dim,
                                                    placeholders=self.placeholders,
                                                    act=self.act,
                                                    dropout=True,
                                                    bias=self.bias,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                # act=self.act,
                                                act=lambda x: x,
                                                dropout=True,
                                                bias=self.bias,
                                                logging=self.logging))

        # self.layers.append(NoisyDense(input_dim=self.hidden_dim,
        #                               units=self.output_dim + 1,
        #                               logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN5_DQN(Model):
    """
    Explicitly input hyperparameters rather than from FLAGS
    Distributed control with binary output at each node
    """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(GCN5_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['actions'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step_tensor, 5000,
                                                                      self.learning_decay, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:,0:self.output_dim])
        diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # diver_loss += self.weight_decay * tf.reduce_mean(self.outputs[:,0:self.output_dim])
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs)
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id == 0:
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            elif layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                hidden = hidden + self.activations[-1]
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        # Opt 1: Plain network
        if self.is_dual:
            self.outputs = tf.reduce_mean(self.activations[-1][:, 0], 0) \
                           + (self.activations[-1][:, 1:] - tf.reduce_mean(self.activations[-1][:, 1:], 0))
        else:
            self.outputs = self.activations[-1]
        # Opt 2: Dueling network
        self.outputs_softmax = tf.nn.softmax(self.outputs, axis=1)
        # self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs_softmax, axis=1)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        # self.layers.append(Dense(input_dim=FLAGS.input_dim,
        #                          output_dim=FLAGS.hidden_dim,
        #                          placeholders=self.placeholders,
        #                          act=tf.nn.leaky_relu,
        #                          bias=self.bias,
        #                          dropout=True,
        #                          logging=self.logging))
        if self.num_layer == 1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                    output_dim=self.hidden_dim,
                                                    placeholders=self.placeholders,
                                                    act=self.act,
                                                    dropout=True,
                                                    bias=self.bias,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                # act=self.act,
                                                act=lambda x: x,
                                                dropout=True,
                                                bias=self.bias,
                                                logging=self.logging))

        self.layers.append(Dense(input_dim=self.hidden_dim,
                                 output_dim=self.hidden_dim,
                                 placeholders=self.placeholders,
                                 act=tf.nn.leaky_relu,
                                 bias=self.bias,
                                 dropout=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=self.hidden_dim,
                                 output_dim=self.hidden_dim,
                                 placeholders=self.placeholders,
                                 act=tf.nn.leaky_relu,
                                 bias=self.bias,
                                 dropout=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=self.hidden_dim,
                                 output_dim=self.hidden_dim,
                                 placeholders=self.placeholders,
                                 act=tf.nn.leaky_relu,
                                 bias=self.bias,
                                 dropout=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=self.hidden_dim,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN3_DQN(Model):
    """Explicitly input hyperparameters rather than from FLAGS """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(GCN3_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step_tensor, 5000,
                                                                      self.learning_decay, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs_softmax[:,0:self.output_dim])
        diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # diver_loss += self.weight_decay * tf.reduce_mean(self.outputs[:,0:self.output_dim])
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs)
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        self.outputs = self.activations[-1]
        self.output_bias = self.outputs[:, 0:2]
        self.output_weight = self.outputs[:, 2:]
        # Opt 2: Fast network
        # self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0)
        self.outputs_softmax = tf.nn.leaky_relu(
            tf.reduce_sum(self.dense_input * self.output_weight[:,2:2+self.input_dim], axis=1, keepdims=True) + self.output_bias[:,0:1])
        self.outputs_softmax = tf.nn.leaky_relu(
            tf.reduce_sum(self.outputs_softmax * self.output_weight[:,2+self.input_dim:], axis=1, keepdims=True) + self.output_bias[:,1:2])
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs_softmax)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if self.num_layer == 1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.input_dim + 1,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                    output_dim=self.hidden_dim,
                                                    placeholders=self.placeholders,
                                                    act=self.act,
                                                    dropout=True,
                                                    bias=self.bias,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=(self.input_dim + 1)*2,
                                                placeholders=self.placeholders,
                                                # act=self.act,
                                                act=lambda x: x,
                                                dropout=True,
                                                bias=self.bias,
                                                logging=self.logging))

        # self.layers.append(NoisyDense(input_dim=self.hidden_dim,
        #                               units=self.output_dim + 1,
        #                               logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN6_DQN(Model):
    """
    Explicitly input hyperparameters rather than from FLAGS
    Distributed control with binary output at each node
    """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(GCN6_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['actions'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step_tensor, 5000,
                                                                      self.learning_decay, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:,0:self.output_dim])
        diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # diver_loss += self.weight_decay * tf.reduce_mean(self.outputs[:,0:self.output_dim])
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs)
        self.loss += diver_loss
        # l0loss = tf.cast(tf.math.count_nonzero(self.outputs_utility), tf.float32)
        # self.loss += 1e-6*l0loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.utilnets:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
            layer_id = layer_id + 1
        self.outputs_utility = self.activations[-1]

        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
            layer_id = layer_id + 1

        # Opt 1: Plain network
        if self.is_dual:
            self.outputs = tf.reduce_mean(self.activations[-1][:, 0], 0) \
                           + (self.activations[-1][:, 1:] - tf.reduce_mean(self.activations[-1][:, 1:], 0))
        else:
            self.outputs = self.activations[-1]
        # Opt 2: Dueling network
        self.outputs_softmax = tf.nn.softmax(self.outputs, axis=1)
        # self.outputs_softmax = self.outputs
        self.pred = tf.argmax(self.outputs_softmax, axis=1)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        self.utilnets.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.leaky_relu,
                                 bias=True,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.utilnets.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))
        if self.num_layer == 1:
            self.layers.append(GraphConvolution(input_dim=1,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=False,
                                                bias=self.bias,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=1,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                dropout=True,
                                                sparse_inputs=False,
                                                bias=self.bias,
                                                logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                    output_dim=self.hidden_dim,
                                                    placeholders=self.placeholders,
                                                    act=self.act,
                                                    dropout=True,
                                                    bias=self.bias,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                bias=self.bias,
                                                logging=self.logging))

        # self.layers.append(NoisyDense(input_dim=self.hidden_dim,
        #                               units=self.output_dim + 1,
        #                               logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GNN_DQN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GNN_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        if FLAGS.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                      initializer=tf.zeros_initializer)
            learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.learning_rate, self.global_step_tensor, 5000,
                                                       FLAGS.learning_decay, staircase=True)
        else:
            learning_rate = FLAGS.learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))

        for i in range(1, FLAGS.diver_num):
            diver_loss = tf.reduce_min([diver_loss,
                                        tf.reduce_mean(
                                            tf.abs(self.outputs[:, i:i + self.output_dim] - self.placeholders['labels']))])
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id > 1 and layer_id < len(self.layers) - 1:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden + self.activations[-2])
                layer_id = layer_id + 1
            elif layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        if not self.skip:
            self.outputs = self.activations[-1]
        else:
            # hiddens = [dense_input] + self.activations[1:]
            hiddens = [self.dense_input] + self.activations[-1:]
            super_hidden = tf.concat(hiddens, axis=1)
            if FLAGS.wts_init == 'random':
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1])
            elif FLAGS.wts_init == 'zeros':
                input_dim = self.dense_input.get_shape().as_list()[1]
                output_dim = self.activations[-1].shape.as_list()[1]
                dense_shape = [super_hidden.get_shape().as_list()[1], output_dim]
                init_wts = np.zeros(dense_shape, dtype=np.float32)
                diag_mtx = np.identity(int(output_dim/2))
                neg_indices = list(range(0, output_dim-1, 2))
                pos_indices = list(range(1, output_dim, 2))
                init_wts[0:int(output_dim/2), neg_indices] = - diag_mtx
                init_wts[0:int(output_dim/2), pos_indices] = diag_mtx
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1], kernel_initializer=tf.constant_initializer(init_wts))

        # self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0)
        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        i_dims = [self.input_dim]
        for i in range(1, FLAGS.num_layer):
            if i % 2==0: # odd layer
                i_dims.append(2*FLAGS.hidden1)
            else: # even layer
                i_dims.append(FLAGS.hidden1)
        if FLAGS.num_layer % 2==0:
            i_dims.append(2*FLAGS.hidden1)
        else:
            i_dims.append(FLAGS.hidden1)
        for i in range(0, FLAGS.num_layer):
            if i % 2 :
                self.layers.append(MaxPoolingAggregator(input_dim=i_dims[i],
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                # act=tf.nn.leaky_relu,
                                                act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=(i==0),
                                                # bias=True,
                                                logging=self.logging))
            else:
                self.layers.append(GraphConvolution(input_dim=i_dims[i],
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.leaky_relu,
                                                # act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=(i==0),
                                                # bias=True,
                                                logging=self.logging))

        self.layers.append(Dense(input_dim=i_dims[-1],
                                 output_dim=FLAGS.diver_num,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class MVGCN_DQN(ModelD):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MVGCN_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.channel_dim = len(placeholders['adj'])
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        if FLAGS.learning_decay < 1.0:
            self.global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[],
                                                      initializer=tf.zeros_initializer)
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step_tensor, 5000,
                                                       FLAGS.learning_decay, staircase=True)
        else:
            learning_rate = FLAGS.learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        diver_loss = tf.sqrt(tf.reduce_mean((self.outputs_softmax - self.placeholders['labels'])**2))
        # self.eligibility = tf.gradients(tf.reduce_mean(self.placeholders['labels'] - self.outputs_softmax),
        #                                 xs=self.state_vec)

        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.dense_input)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        if self.name == 'mvgcn_dqn':
            self.attacts.append(tf.concat([self.activations[-2], self.dense_input], axis=-1))
            for ilayer in range(0, len(self.utilnets)):
                layer = self.utilnets[ilayer]
                hidden = layer(self.attacts[-1])
                self.attacts.append(hidden)

        if not self.skip:
            self.outputs = self.activations[-1]
        else:
            # hiddens = [dense_input] + self.activations[1:]
            hiddens = [self.dense_input] + self.activations[-1:]
            super_hidden = tf.concat(hiddens, axis=1)
            if FLAGS.wts_init == 'random':
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1])
            elif FLAGS.wts_init == 'zeros':
                input_dim = self.dense_input.get_shape().as_list()[1]
                output_dim = self.activations[-1].shape.as_list()[1]
                dense_shape = [super_hidden.get_shape().as_list()[1], output_dim]
                init_wts = np.zeros(dense_shape, dtype=np.float32)
                diag_mtx = np.identity(int(output_dim/2))
                neg_indices = list(range(0, output_dim-1, 2))
                pos_indices = list(range(1, output_dim, 2))
                init_wts[0:int(output_dim/2), neg_indices] = - diag_mtx
                init_wts[0:int(output_dim/2), pos_indices] = diag_mtx
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1], kernel_initializer=tf.constant_initializer(init_wts))

        self.outputs_softmax = tf.multiply(self.activations[-1], self.attacts[-1])
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs_softmax)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        i_dims = [self.input_dim]
        for i in range(1, FLAGS.num_layer):
            i_dims.append(FLAGS.hidden1)
        i_dims.append(FLAGS.hidden1)
        for i in range(0, FLAGS.num_layer):
            self.layers.append(MVGraphConvolution(input_dim=i_dims[i],
                                            output_dim=FLAGS.hidden1,
                                            channel_dim=self.channel_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            # act=lambda x: x,
                                            dropout=True,
                                            sparse_inputs=False,
                                            aggregator='clique',
                                            transpose=(i == FLAGS.num_layer-1),
                                            bias=True,
                                            logging=self.logging))

        self.layers.append(Dense(input_dim=i_dims[-1],
                                 output_dim=FLAGS.diver_num,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))
        # self.utilnets.append(Dense(input_dim=(i_dims[-1] + self.input_dim) * self.channel_dim,
        #                          output_dim=FLAGS.hidden1 * self.channel_dim,
        #                          placeholders=self.placeholders,
        #                          act=tf.nn.leaky_relu,
        #                          dropout=True,
        #                          logging=self.logging))
        # self.utilnets.append(MaxGate(input_dim=FLAGS.hidden1 * self.channel_dim,
        #                            sparse_inputs=False,
        #                            act=lambda x: x,
        #                            rank=2,
        #                            logging=self.logging))
        # self.utilnets.append(Dense(input_dim=FLAGS.hidden1 * self.channel_dim,
        #                          output_dim=FLAGS.diver_num * self.channel_dim,
        #                          placeholders=self.placeholders,
        #                          act=lambda x: x,
        #                          dropout=True,
        #                          logging=self.logging))


    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class MVGCN2_DQN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MVGCN2_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.channel_dim = len(placeholders['adj'])
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        if FLAGS.learning_decay < 1.0:
            self.global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[],
                                                      initializer=tf.zeros_initializer)
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step_tensor, 5000,
                                                       FLAGS.learning_decay, staircase=True)
        else:
            learning_rate = FLAGS.learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        diver_loss = tf.sqrt(tf.reduce_mean((self.outputs_softmax - self.placeholders['labels'])**2))

        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        i_dims = [self.input_dim]
        for i in range(1, FLAGS.num_layer):
            i_dims.append(FLAGS.hidden1)
        i_dims.append(FLAGS.hidden1 * self.channel_dim)
        for ic in range(0, self.channel_dim):
            self.layers.append(GraphConvolution(input_dim=i_dims[0],
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.leaky_relu,
                                                # act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=True,
                                                logging=self.logging,
                                                channel=ic,
                                                num_channels=self.channel_dim))

        self.layers.append(Dense(input_dim=i_dims[-1],
                                 output_dim=FLAGS.diver_num,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))
        self.utilnets.append(Dense(input_dim=i_dims[-1]+self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.leaky_relu,
                                 dropout=True,
                                 logging=self.logging))
        self.utilnets.append(MaxGate(input_dim=FLAGS.hidden1,
                                   sparse_inputs=False,
                                   act=lambda x: x,
                                   rank=2,
                                   logging=self.logging))
        self.utilnets.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=FLAGS.diver_num,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))


    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN_DQN_KF(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_DQN_KF, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        if FLAGS.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                      initializer=tf.zeros_initializer)
            learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.learning_rate, self.global_step_tensor, 5000,
                                                       FLAGS.learning_decay, staircase=True)
        else:
            learning_rate = FLAGS.learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        # for var in self.layers[0].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        greedy_util_est = self.placeholders['greedy'] * self.outputs_utility
        util_loss = tf.abs(self.placeholders['reward'] - tf.reduce_sum(greedy_util_est))
        diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs_softmax[:,0:self.output_dim] - self.placeholders['labels'])**2))
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs[:,0:self.output_dim])

        for i in range(1, FLAGS.diver_num):
            diver_loss = tf.reduce_min([diver_loss,
                                        tf.reduce_mean(
                                            tf.abs(self.outputs[:, i:i + self.output_dim] - self.placeholders['labels']))])
        self.loss += diver_loss
        self.loss_crt = util_loss

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.normed_input)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        self.attacts.append(self.dense_input)
        for ilayer in range(0, len(self.utilnets)):
            layer = self.utilnets[ilayer]
            hidden = layer(self.attacts[-1])
            self.attacts.append(hidden)

        if not self.skip:
            self.outputs = self.activations[-1]
        else:
            # hiddens = [dense_input] + self.activations[1:]
            hiddens = [self.dense_input] + self.activations[-1:]
            super_hidden = tf.concat(hiddens, axis=1)
            if FLAGS.wts_init == 'random':
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1])
            elif FLAGS.wts_init == 'zeros':
                input_dim = self.dense_input.get_shape().as_list()[1]
                output_dim = self.activations[-1].shape.as_list()[1]
                dense_shape = [super_hidden.get_shape().as_list()[1], output_dim]
                init_wts = np.zeros(dense_shape, dtype=np.float32)
                diag_mtx = np.identity(int(output_dim/2))
                neg_indices = list(range(0, output_dim-1, 2))
                pos_indices = list(range(1, output_dim, 2))
                init_wts[0:int(output_dim/2), neg_indices] = - diag_mtx
                init_wts[0:int(output_dim/2), pos_indices] = diag_mtx
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1], kernel_initializer=tf.constant_initializer(init_wts))

        self.outputs_softmax = tf.multiply(self.activations[-1], self.attacts[-1])
        self.outputs_utility = self.attacts[-1]
        self.pred = tf.argmax(self.outputs_softmax)


    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
            self.opt_op_crt = self.optimizer_crt.minimize(self.loss_crt, global_step=self.global_step_tensor)
        else:
            self.opt_op = self.optimizer.minimize(self.loss)
            self.opt_op_crt = self.optimizer_crt.minimize(self.loss_crt)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if FLAGS.num_layer==1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.diver_num,
                                                placeholders=self.placeholders,
                                                act=tf.nn.leaky_relu,
                                                # act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=False,
                                                # bias=True,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                # act=tf.nn.leaky_relu,
                                                act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=False,
                                                logging=self.logging))
            for i in range(FLAGS.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                    output_dim=FLAGS.hidden1,
                                                    placeholders=self.placeholders,
                                                    # act=tf.nn.leaky_relu,
                                                    act=lambda x: x,
                                                    dropout=True,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.diver_num,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                logging=self.logging))

        self.utilnets.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.leaky_relu,
                                 # act=lambda x: x,
                                 bias=True,
                                 dropout=False,
                                 logging=self.logging))

        for i in range(10):
            self.utilnets.append(Dense(input_dim=FLAGS.hidden1,
                                     output_dim=FLAGS.hidden1,
                                     placeholders=self.placeholders,
                                     act=tf.nn.leaky_relu,
                                     # act=lambda x: x,
                                     bias=True,
                                     dropout=False,
                                     logging=self.logging))

        self.utilnets.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=FLAGS.diver_num,
                                 placeholders=self.placeholders,
                                 act=tf.nn.leaky_relu,
                                 # act=lambda x: x,
                                 bias=True,
                                 dropout=False,
                                 logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN_DQN_LA(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_DQN_LA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        if FLAGS.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                      initializer=tf.zeros_initializer)
            learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.learning_rate, self.global_step_tensor, 5000,
                                                       FLAGS.learning_decay, staircase=True)
        else:
            learning_rate = FLAGS.learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        vars = tf.compat.v1.trainable_variables(scope=self.name)
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                           if 'bias' not in v.name])
        self.loss += FLAGS.weight_decay * lossL2

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs[:,0:self.output_dim])

        for i in range(1, FLAGS.diver_num):
            diver_loss = tf.reduce_min([diver_loss,
                                        tf.reduce_mean(
                                            tf.abs(self.outputs[:, i:i + self.output_dim] - self.placeholders['labels']))])
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.normed_input)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        if not self.skip:
            self.outputs = self.activations[-1]
        else:
            # hiddens = [dense_input] + self.activations[1:]
            hiddens = [self.dense_input] + self.activations[-1:]
            super_hidden = tf.concat(hiddens, axis=1)
            if FLAGS.wts_init == 'random':
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1])
            elif FLAGS.wts_init == 'zeros':
                input_dim = self.dense_input.get_shape().as_list()[1]
                output_dim = self.activations[-1].shape.as_list()[1]
                dense_shape = [super_hidden.get_shape().as_list()[1], output_dim]
                init_wts = np.zeros(dense_shape, dtype=np.float32)
                diag_mtx = np.identity(int(output_dim/2))
                neg_indices = list(range(0, output_dim-1, 2))
                pos_indices = list(range(1, output_dim, 2))
                init_wts[0:int(output_dim/2), neg_indices] = - diag_mtx
                init_wts[0:int(output_dim/2), pos_indices] = diag_mtx
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1], kernel_initializer=tf.constant_initializer(init_wts))

        if self.name == 'gcn_dqn_la':
            self.outputs_softmax = tf.multiply(self.activations[-1], self.dense_input)
            self.outputs_utility = self.dense_input
            self.pred = tf.argmax(self.outputs_softmax)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if FLAGS.num_layer == 1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.diver_num,
                                                placeholders=self.placeholders,
                                                act=tf.nn.leaky_relu,
                                                # act=lambda x: x,
                                                dropout=True,
                                                # sparse_inputs=True,
                                                # bias=True,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.leaky_relu,
                                                # act=lambda x: x,
                                                dropout=True,
                                                # sparse_inputs=True,
                                                logging=self.logging))
            for i in range(FLAGS.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                    output_dim=FLAGS.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.leaky_relu,
                                                    # act=lambda x: x,
                                                    dropout=True,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.diver_num,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                logging=self.logging))


    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN_CRL(Model):
    '''
    GCN-based Constrained Reinforcement Learning
    '''
    def __init__(self, placeholders, flags, **kwargs):
        super(GCN_CRL, self).__init__(**kwargs)

        self.flags = flags
        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.batchsize = placeholders['features'].get_shape().as_list()[0]
        self.inputs_action = placeholders["actions"]
        # self.inputs_action = tf.reshape(placeholders["actions"], (-1, self.output_dim))
        self.placeholders = placeholders

        if self.flags.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            learning_rate = tf.compat.v1.train.exponential_decay(self.flags.learning_rate,
                                                                 self.global_step_tensor,
                                                                 5000,
                                                                 self.flags.learning_decay, staircase=True)
        else:
            learning_rate = self.flags.learning_rate
        self.opt_act = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.opt_crt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        l2loss_act = 0
        for var in self.actor_layers[0].vars.values():
            l2loss_act += self.flags.weight_decay * tf.nn.l2_loss(var)
        self.loss_act += l2loss_act
        l2loss_crt = 0
        for var in self.critic_layers[0].vars.values():
            l2loss_crt += self.flags.weight_decay * tf.nn.l2_loss(var)
        self.loss_crt += l2loss_crt

        # regression loss
        loss_crt = tf.sqrt(tf.reduce_mean((self.placeholders['labels'] - self.critic_pred)**2))
        # mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.critic_pred)
        # loss_mse = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # loss_mse = tf.sqrt(tf.reduce_mean((self.placeholders['labels']-self.critic_pred)**2))
        # loss_crt = tf.sqrt(tf.reduce_mean((self.placeholders['network_q']-self.critic_q)**2))
        loss_act = -tf.reduce_sum(self.critic_pred) # / (tf.cast(tf.math.count_nonzero(self.placeholders['labels']), tf.float32)+1e-6)
        # loss_act = -tf.reduce_mean(self.critic_q)
        # loss_act = tf.sqrt(tf.reduce_mean((self.placeholders['labels']-self.actor_pred)**2))
        self.act_vars = tf.compat.v1.trainable_variables(scope=self.name+'/'+'actor')
        self.crt_vars = tf.compat.v1.trainable_variables(scope=self.name+'/'+'critic')
        self.loss_act += loss_act
        self.loss_crt += loss_crt

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        # actor_inputs = tf.concat([self.activations[-1], dense_input], axis=-1)
        self.actors.append(self.inputs)
        for ilayer in range(0, len(self.actor_layers)):
            layer = self.actor_layers[ilayer]
            hidden = layer(self.actors[-1])
            self.actors.append(hidden)
        self.actor_pred = self.actors[-1]
        selector = self.placeholders['labels_mask'] * tf.ones_like(self.placeholders['labels'], dtype=tf.int32)
        selector = tf.cast(selector, tf.bool)
        self.critic_actions = tf.where(selector, self.actor_pred, self.inputs_action)
        critic_inputs = tf.concat([self.dense_input, self.critic_actions], axis=-1)
        # critic_inputs = tf.concat([self.dense_input, self.actor_pred], axis=-1)
        # critic_inputs = self.inputs_critic
        self.critics.append(critic_inputs)
        for ilayer in range(0, len(self.critic_layers)):
            layer = self.critic_layers[ilayer]
            hidden = layer(self.critics[-1])
            self.critics.append(hidden)
        self.critic_pred = self.critics[-1][:, 0]
        self.network_weight = tf.nn.softmax(self.critics[-1][:, 1], axis=0)
        self.critic_q = tf.reduce_sum(self.critic_pred * self.network_weight)

        self.outputs = self.actor_pred
        self.actions = self.actor_pred
        self.qvalues = self.critic_pred
        self.outputs_softmax = self.actions # just to be compatable

    def _opt_set(self):
        if self.flags.learning_decay < 1.0:
            self.opt_act_op = self.opt_act.minimize(self.loss_act, var_list=self.act_vars, global_step=self.global_step_tensor)
            self.opt_crt_op = self.opt_crt.minimize(self.loss_crt, var_list=self.crt_vars, global_step=self.global_step_tensor)
        else:
            # gvs = self.opt_act.compute_gradients(self.loss_act, var_list=self.act_vars)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # self.opt_act_op = self.opt_act.apply_gradients(capped_gvs)
            # gvs = self.opt_crt.compute_gradients(self.loss_crt, var_list=self.crt_vars)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # self.opt_crt_op = self.opt_crt.apply_gradients(capped_gvs)
            self.opt_act_op = self.opt_act.minimize(self.loss_act, var_list=self.act_vars)
            self.opt_crt_op = self.opt_crt.minimize(self.loss_crt, var_list=self.crt_vars)
            # grads_and_vars = self.opt_crt.compute_gradients(self.loss_crt, self.inputs.vars )

    def _build(self):
        with tf.compat.v1.variable_scope('actor'):
            _LAYER_UIDS['actor'] = 0
            if self.flags.num_layer == 1:
                self.actor_layers.append(GraphConvolution(input_dim=self.input_dim,
                                                    output_dim=self.flags.diver_num,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.relu,
                                                    # act=lambda x: x,
                                                    # act=tf.nn.relu6,
                                                    dropout=True,
                                                    sparse_inputs=True,
                                                    # bias=True,
                                                    logging=self.logging))
            else:
                self.actor_layers.append(GraphConvolution(input_dim=self.input_dim,
                                                    output_dim=self.flags.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.leaky_relu,
                                                    # act=tf.nn.relu6,
                                                    # act=lambda x: x,
                                                    dropout=True,
                                                    sparse_inputs=True,
                                                    logging=self.logging))
                for i in range(self.flags.num_layer - 2):
                    self.actor_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                        output_dim=self.flags.hidden1,
                                                        placeholders=self.placeholders,
                                                        act=tf.nn.leaky_relu,
                                                        # act=tf.nn.relu6,
                                                        # act=lambda x: x,
                                                        dropout=True,
                                                        logging=self.logging))

                self.actor_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                          output_dim=self.flags.diver_num,
                                                          placeholders=self.placeholders,
                                                          act=tf.nn.relu,
                                                          # act=lambda x: x,
                                                          # act=tf.nn.relu6,
                                                          dropout=True,
                                                          logging=self.logging))

            # self.actor_layers.append(Dense(input_dim=self.flags.hidden1,
            #                          output_dim=self.flags.diver_num,
            #                          placeholders=self.placeholders,
            #                          act=lambda x: x,
            #                          dropout=True,
            #                          logging=self.logging))

        with tf.compat.v1.variable_scope('critic'):
            _LAYER_UIDS['critic'] = 0
            if self.flags.num_layer == 1:
                self.critic_layers.append(GraphConvolution(input_dim=self.input_dim+self.flags.diver_num,
                                                    output_dim=self.flags.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.leaky_relu,
                                                    # act=lambda x: x,
                                                    # act=tf.nn.relu6,
                                                    dropout=True,
                                                    bias=True,
                                                    logging=self.logging))
            else:
                self.critic_layers.append(GraphConvolution(input_dim=self.input_dim+self.flags.diver_num,
                                                    output_dim=self.flags.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.leaky_relu,
                                                    # act=tf.nn.relu6,
                                                    # act=lambda x: x,
                                                    dropout=True,
                                                           bias=True,
                                                           logging=self.logging))
                for i in range(self.flags.num_layer - 2):
                    self.critic_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                        output_dim=self.flags.hidden1,
                                                        placeholders=self.placeholders,
                                                        act=tf.nn.leaky_relu,
                                                        # act=tf.nn.relu6,
                                                        dropout=True,
                                                               bias=True,
                                                               logging=self.logging))

                self.critic_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                    output_dim=self.flags.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.leaky_relu,
                                                    # act=tf.nn.relu6,
                                                    dropout=True,
                                                           bias=True,
                                                           logging=self.logging))
            # Critic: create refined action
            self.critic_layers.append(Dense(input_dim=self.flags.hidden1,
                                            output_dim=2,
                                            placeholders=self.placeholders,
                                            # act=tf.nn.leaky_relu,
                                            act=lambda x: x,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN_VPG(Model):
    """Explicitly input hyperparameters rather than from FLAGS """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(GCN_VPG, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step_tensor, 5000,
                                                                      self.learning_decay, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:, 0:self.output_dim])
        # diver_loss = tf.sqrt(self.placeholders['network_q'] * tf.reduce_mean(mse, name="loss"))
        # 'network_q' is the return value
        diver_loss = self.placeholders['network_q'] * tf.reduce_mean(mse, name="loss")
        # diver_loss = tf.reduce_mean(mse, name="loss")
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
            layer_id = layer_id + 1

        # Opt 1: Plain network
        self.outputs = self.activations[-1]
        # Opt 2: Dueling network
        # self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0)
        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if self.num_layer == 1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                    output_dim=self.hidden_dim,
                                                    placeholders=self.placeholders,
                                                    act=self.act,
                                                    dropout=True,
                                                    bias=self.bias,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                bias=self.bias,
                                                logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN_FOO(Model):
    """Explicitly input hyperparameters rather than from FLAGS """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(GCN_FOO, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step_tensor, 5000,
                                                                      self.learning_decay, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        mse = tf.compat.v1.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:, 0:self.output_dim],
                                           reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE)
        # diver_loss = 0.5*tf.reduce_mean(mse, name="loss")
        diver_loss = 0.5*mse
        # diver_loss += self.weight_decay * tf.reduce_mean(self.outputs[:,0:self.output_dim])
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs)
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        # Opt 1: Plain network
        if self.is_dual:
            self.outputs = tf.reduce_mean(self.activations[-1][:, 0], 0) \
                           + (self.activations[-1][:, 1:] - tf.reduce_mean(self.activations[-1][:, 1:], 0))
        else:
            self.outputs = self.activations[-1]
        # Opt 2: Dueling network
        # self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0)
        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if self.num_layer == 1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                    output_dim=self.hidden_dim,
                                                    placeholders=self.placeholders,
                                                    act=self.act,
                                                    dropout=True,
                                                    bias=self.bias,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                bias=self.bias,
                                                logging=self.logging))

        # self.layers.append(NoisyDense(input_dim=self.hidden_dim,
        #                               units=self.output_dim + 1,
        #                               logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN_Critic(Model):
    """Explicitly input hyperparameters rather than from FLAGS """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(GCN_Critic, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step_tensor, 5000,
                                                                      self.learning_decay, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:, 0:self.output_dim])
        diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # diver_loss += self.weight_decay * tf.reduce_mean(self.outputs[:,0:self.output_dim])
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs)
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        # Opt 1: Plain network
        if self.is_dual:
            self.outputs = tf.reduce_mean(self.activations[-1][:, 0], 0) \
                           + (self.activations[-1][:, 1:] - tf.reduce_mean(self.activations[-1][:, 1:], 0))
        else:
            self.outputs = self.activations[-1]
        # Opt 2: Dueling network
        # self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0)
        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if self.num_layer == 1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=tf.nn.leaky_relu,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                    output_dim=self.hidden_dim,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.leaky_relu,
                                                    dropout=True,
                                                    bias=self.bias,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                bias=self.bias,
                                                logging=self.logging))

        # self.layers.append(NoisyDense(input_dim=self.hidden_dim,
        #                               units=self.output_dim + 1,
        #                               logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN_A2C(Model):
    '''
    GCN-based Constrained Reinforcement Learning
    '''
    def __init__(self, placeholders, flags, **kwargs):
        super(GCN_A2C, self).__init__(**kwargs)

        self.flags = flags
        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.batchsize = placeholders['features'].get_shape().as_list()[0]
        self.inputs_action = placeholders["actions"]
        # self.inputs_action = tf.reshape(placeholders["actions"], (-1, self.output_dim))
        self.placeholders = placeholders

        if self.flags.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            learning_rate = tf.compat.v1.train.exponential_decay(self.flags.learning_rate,
                                                                 self.global_step_tensor,
                                                                 5000,
                                                                 self.flags.learning_decay, staircase=True)
        else:
            learning_rate = self.flags.learning_rate
        self.opt_act = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.opt_crt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        l2loss_act = 0
        for var in self.actor_layers[0].vars.values():
            l2loss_act += self.flags.weight_decay * tf.nn.l2_loss(var)
        self.loss_act += l2loss_act
        l2loss_crt = 0
        for var in self.critic_layers[0].vars.values():
            l2loss_crt += self.flags.weight_decay * tf.nn.l2_loss(var)
        self.loss_crt += l2loss_crt

        # regression loss
        loss_crt = tf.sqrt(tf.reduce_mean((self.placeholders['labels'] - self.critic_pred)**2))
        # mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.critic_pred)
        # loss_mse = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # loss_mse = tf.sqrt(tf.reduce_mean((self.placeholders['labels']-self.critic_pred)**2))
        # loss_crt = tf.sqrt(tf.reduce_mean((self.placeholders['network_q']-self.critic_q)**2))
        # loss_act = -tf.reduce_sum(self.critic_pred) # / (tf.cast(tf.math.count_nonzero(self.placeholders['labels']), tf.float32)+1e-6)
        loss_act = -self.critic_q
        # loss_act = -tf.reduce_mean(self.critic_q)
        # loss_act = tf.sqrt(tf.reduce_mean((self.placeholders['labels']-self.actor_pred)**2))
        self.act_vars = tf.compat.v1.trainable_variables(scope=self.name+'/'+'actor')
        self.crt_vars = tf.compat.v1.trainable_variables(scope=self.name+'/'+'critic')
        self.loss_act += loss_act
        self.loss_crt += loss_crt

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        # actor_inputs = tf.concat([self.activations[-1], dense_input], axis=-1)
        self.actors.append(self.inputs)
        for ilayer in range(0, len(self.actor_layers)):
            layer = self.actor_layers[ilayer]
            hidden = layer(self.actors[-1])
            self.actors.append(hidden)
        self.actor_pred = self.actors[-1]
        selector = self.placeholders['labels_mask'] * tf.ones_like(self.placeholders['labels'], dtype=tf.int32)
        selector = tf.cast(selector, tf.bool)
        self.critic_actions = tf.where(selector, self.actor_pred, self.inputs_action)
        critic_inputs = tf.concat([self.dense_input, self.critic_actions], axis=-1)
        # critic_inputs = tf.concat([self.dense_input, self.actor_pred], axis=-1)
        # critic_inputs = self.inputs_critic
        self.critics.append(critic_inputs)
        for ilayer in range(0, len(self.critic_layers)):
            layer = self.critic_layers[ilayer]
            hidden = layer(self.critics[-1])
            self.critics.append(hidden)
        self.critic_pred = self.critics[-1][:, 0]
        self.network_weight = tf.nn.softmax(self.critics[-1][:, 1], axis=0)
        self.critic_q = tf.reduce_sum(self.critic_pred * self.network_weight)

        self.outputs = self.actor_pred
        self.actions = self.actor_pred
        self.qvalues = self.critic_pred
        self.outputs_softmax = self.actions # just to be compatable

    def _opt_set(self):
        if self.flags.learning_decay < 1.0:
            self.opt_act_op = self.opt_act.minimize(self.loss_act, var_list=self.act_vars, global_step=self.global_step_tensor)
            self.opt_crt_op = self.opt_crt.minimize(self.loss_crt, var_list=self.crt_vars, global_step=self.global_step_tensor)
        else:
            # gvs = self.opt_act.compute_gradients(self.loss_act, var_list=self.act_vars)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # self.opt_act_op = self.opt_act.apply_gradients(capped_gvs)
            # gvs = self.opt_crt.compute_gradients(self.loss_crt, var_list=self.crt_vars)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # self.opt_crt_op = self.opt_crt.apply_gradients(capped_gvs)
            self.opt_act_op = self.opt_act.minimize(self.loss_act, var_list=self.act_vars)
            self.opt_crt_op = self.opt_crt.minimize(self.loss_crt, var_list=self.crt_vars)
            # grads_and_vars = self.opt_crt.compute_gradients(self.loss_crt, self.inputs.vars )

    def _build(self):
        with tf.compat.v1.variable_scope('actor'):
            _LAYER_UIDS['actor'] = 0
            if self.flags.num_layer == 1:
                self.actor_layers.append(GraphConvolution(input_dim=self.input_dim,
                                                    output_dim=self.flags.diver_num,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.relu,
                                                    # act=lambda x: x,
                                                    # act=tf.nn.relu6,
                                                    dropout=True,
                                                    sparse_inputs=True,
                                                    # bias=True,
                                                    logging=self.logging))
            else:
                self.actor_layers.append(GraphConvolution(input_dim=self.input_dim,
                                                    output_dim=self.flags.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.leaky_relu,
                                                    # act=tf.nn.relu6,
                                                    # act=lambda x: x,
                                                    dropout=True,
                                                    sparse_inputs=True,
                                                    logging=self.logging))
                for i in range(self.flags.num_layer - 2):
                    self.actor_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                        output_dim=self.flags.hidden1,
                                                        placeholders=self.placeholders,
                                                        act=tf.nn.leaky_relu,
                                                        # act=tf.nn.relu6,
                                                        # act=lambda x: x,
                                                        dropout=True,
                                                        logging=self.logging))

                self.actor_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                          output_dim=self.flags.diver_num,
                                                          placeholders=self.placeholders,
                                                          act=tf.nn.relu,
                                                          # act=lambda x: x,
                                                          # act=tf.nn.relu6,
                                                          dropout=True,
                                                          logging=self.logging))

            # self.actor_layers.append(Dense(input_dim=self.flags.hidden1,
            #                          output_dim=self.flags.diver_num,
            #                          placeholders=self.placeholders,
            #                          act=lambda x: x,
            #                          dropout=True,
            #                          logging=self.logging))

        with tf.compat.v1.variable_scope('critic'):
            _LAYER_UIDS['critic'] = 0
            if self.flags.num_layer == 1:
                self.critic_layers.append(GraphConvolution(input_dim=self.input_dim+self.flags.diver_num,
                                                    output_dim=self.flags.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.leaky_relu,
                                                    # act=lambda x: x,
                                                    # act=tf.nn.relu6,
                                                    dropout=True,
                                                    bias=True,
                                                    logging=self.logging))
            else:
                self.critic_layers.append(GraphConvolution(input_dim=self.input_dim+self.flags.diver_num,
                                                    output_dim=self.flags.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.leaky_relu,
                                                    # act=tf.nn.relu6,
                                                    # act=lambda x: x,
                                                    dropout=True,
                                                           bias=True,
                                                           logging=self.logging))
                for i in range(self.flags.num_layer - 2):
                    self.critic_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                        output_dim=self.flags.hidden1,
                                                        placeholders=self.placeholders,
                                                        act=tf.nn.leaky_relu,
                                                        # act=tf.nn.relu6,
                                                        dropout=True,
                                                               bias=True,
                                                               logging=self.logging))

                self.critic_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                    output_dim=self.flags.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.leaky_relu,
                                                    # act=tf.nn.relu6,
                                                    dropout=True,
                                                           bias=True,
                                                           logging=self.logging))
            # Critic: create refined action
            self.critic_layers.append(Dense(input_dim=self.flags.hidden1,
                                            output_dim=2,
                                            placeholders=self.placeholders,
                                            # act=tf.nn.leaky_relu,
                                            act=lambda x: x,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN_Dense(ModelD):
    '''
    GCN-based Constrained Reinforcement Learning
    '''
    def __init__(self, placeholders, flags,
                 act=tf.nn.leaky_relu, act_output=lambda x:x, num_layer=1, 
                 bias=False, dropout=False, is_dual=False, is_noisy=False,
                 **kwargs):
        super(GCN_Dense, self).__init__(**kwargs)

        self.flags = flags
        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.batchsize = placeholders['features'].get_shape().as_list()[0]
        # self.inputs_action = placeholders["actions"]
        # self.inputs_action = tf.reshape(placeholders["actions"], (-1, self.output_dim))
        self.placeholders = placeholders
        self.act = act
        self.act_output = act_output
        self.num_layer = num_layer
        self.bias = bias
        self.dropout = dropout
        self.is_dual = is_dual
        self.is_noisy = is_noisy

        if self.flags.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(  self.flags.learning_rate,
                                                                        self.global_step_tensor,
                                                                        5000,
                                                                        self.flags.learning_decay, 
                                                                        staircase=True)
        else:
            self.learning_rate = self.flags.learning_rate
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        l2loss = 0
        for var in self.layers[0].vars.values():
            l2loss += self.flags.weight_decay * tf.nn.l2_loss(var)
        self.loss += l2loss
        # regression loss
        loss_mse = tf.sqrt(tf.reduce_mean((self.placeholders['labels'] - self.outputs)**2))
        # mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs)
        # loss_mse = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        self.vars = tf.compat.v1.trainable_variables(scope=self.name)
        self.loss += loss_mse

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        self.activations.append(self.inputs)
        for ilayer in range(0, len(self.layers)):
            layer = self.layers[ilayer]
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0) # just to be compatable            

    def _opt_set(self):
        if self.flags.learning_decay < 1.0:
            self.opt_op = self.opt.minimize(self.loss, var_list=self.vars, global_step=self.global_step_tensor)
        else:
            self.opt_op = self.opt.minimize(self.loss, var_list=self.vars)

    def _build(self):
        with tf.compat.v1.variable_scope(self.name):
            _LAYER_UIDS['layer'] = 0
            if self.flags.num_layer == 1:
                self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                    output_dim=self.flags.diver_num,
                                                    placeholders=self.placeholders,
                                                    act=self.act_output,
                                                    dropout=self.dropout,
                                                    sparse_inputs=False,
                                                    bias=self.bias,
                                                    logging=self.logging))
            else:
                self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                    output_dim=self.flags.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=self.act,
                                                    dropout=self.dropout,
                                                    sparse_inputs=False,
                                                    bias=self.bias,
                                                    logging=self.logging))
                for i in range(self.flags.num_layer - 2):
                    self.layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                        output_dim=self.flags.hidden1,
                                                        placeholders=self.placeholders,
                                                        act=self.act,
                                                        dropout=self.dropout,
                                                        bias=self.bias,
                                                        logging=self.logging))

                self.layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                    output_dim=self.flags.diver_num,
                                                    placeholders=self.placeholders,
                                                    act=self.act_output,
                                                    dropout=self.dropout,
                                                    bias=self.bias,
                                                    logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs

    def call(self, inputs):
        # return tf.nn.softmax(self.outputs, axis=0)
        self.placeholders = inputs
        self.build()
        return self.outputs        


class GNN_A2C(Model):
    '''
    GCN-based Constrained Reinforcement Learning
    '''
    def __init__(self, placeholders, flags, **kwargs):
        super(GNN_A2C, self).__init__(**kwargs)

        self.flags = flags
        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.batchsize = placeholders['features'].get_shape().as_list()[0]
        self.inputs_action = placeholders["actions"]
        self.hidden_inputs = placeholders["hidden"]
        # self.inputs_action = tf.reshape(placeholders["actions"], (-1, self.output_dim))
        self.placeholders = placeholders

        if self.flags.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            learning_rate = tf.compat.v1.train.exponential_decay(self.flags.learning_rate,
                                                                 self.global_step_tensor,
                                                                 5000,
                                                                 self.flags.learning_decay, staircase=True)
        else:
            learning_rate = self.flags.learning_rate
        self.opt_act = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.opt_crt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.signal_layers = []
        self.state_layers = []
        self.signals = []
        self.states = []

        self.build()

    def _loss(self):
        # Weight decay loss
        l2loss_act = 0
        for var in self.actor_layers[0].vars.values():
            l2loss_act += self.flags.weight_decay * tf.nn.l2_loss(var)
        self.loss_act += l2loss_act
        l2loss_crt = 0
        for var in self.critic_layers[0].vars.values():
            l2loss_crt += self.flags.weight_decay * tf.nn.l2_loss(var)
        self.loss_crt += l2loss_crt

        # regression loss
        loss_crt = tf.sqrt(tf.reduce_mean((self.placeholders['labels'] - self.critic_pred)**2))
        # mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.critic_pred)
        # loss_mse = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # loss_mse = tf.sqrt(tf.reduce_mean((self.placeholders['labels']-self.critic_pred)**2))
        # loss_crt = tf.sqrt(tf.reduce_mean((self.placeholders['network_q']-self.critic_q)**2))
        loss_act = -tf.reduce_mean(self.critic_pred) # / (tf.cast(tf.math.count_nonzero(self.placeholders['labels']), tf.float32)+1e-6)
        # loss_act = -tf.reduce_mean(self.critic_q)
        # loss_act = tf.sqrt(tf.reduce_mean((self.placeholders['labels']-self.actor_pred)**2))
        self.enc_vars = tf.compat.v1.trainable_variables(scope=self.name+'/'+'encoder')
        self.act_vars = tf.compat.v1.trainable_variables(scope=self.name+'/'+'actor')
        self.crt_vars = tf.compat.v1.trainable_variables(scope=self.name+'/'+'critic')
        self.act_vars += self.enc_vars
        self.crt_vars += self.enc_vars
        self.loss_act += loss_act
        self.loss_crt += loss_crt

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # signal path
        self.signals.append(self.inputs)
        for ilayer in range(0, len(self.signal_layers)):
            layer = self.signal_layers[ilayer]
            hidden = layer(self.signals[-1])
            self.signals.append(hidden)
        signal_out = self.signals[-1]
        self.hidden_out = signal_out

        selector = self.placeholders['labels_mask'] * tf.ones_like(self.placeholders['labels'], dtype=tf.int32)
        selector = tf.cast(selector, tf.bool)
        self.actors.append(self.hidden_out)
        for ilayer in range(0, len(self.actor_layers)):
            layer = self.actor_layers[ilayer]
            hidden = layer(self.actors[-1])
            self.actors.append(hidden)
        self.actor_pred = self.actors[-1]
        self.critic_actions = tf.where(selector, self.actor_pred, self.inputs_action)
        state_inputs = signal_out
        critic_inputs = tf.concat([state_inputs, self.critic_actions], axis=-1)
        self.critics.append(critic_inputs)
        for ilayer in range(0, len(self.critic_layers)):
            layer = self.critic_layers[ilayer]
            hidden = layer(self.critics[-1])
            self.critics.append(hidden)
        self.critic_pred = self.critics[-1][:, 0]
        self.network_weight = tf.nn.softmax(self.critics[-1][:, 1], axis=0)
        self.critic_q = tf.reduce_sum(self.critic_pred * self.network_weight)
        self.critic_q = self.critic_pred

        self.outputs = self.actor_pred
        self.actions = self.actor_pred
        self.qvalues = self.critic_pred
        self.outputs_softmax = self.actions # just to be compatable

    def _opt_set(self):
        if self.flags.learning_decay < 1.0:
            self.opt_act_op = self.opt_act.minimize(self.loss_act, var_list=self.act_vars, global_step=self.global_step_tensor)
            self.opt_crt_op = self.opt_crt.minimize(self.loss_crt, var_list=self.crt_vars, global_step=self.global_step_tensor)
        else:
            # gvs = self.opt_act.compute_gradients(self.loss_act, var_list=self.act_vars)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # self.opt_act_op = self.opt_act.apply_gradients(capped_gvs)
            # gvs = self.opt_crt.compute_gradients(self.loss_crt, var_list=self.crt_vars)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # self.opt_crt_op = self.opt_crt.apply_gradients(capped_gvs)
            self.opt_act_op = self.opt_act.minimize(self.loss_act, var_list=self.act_vars)
            self.opt_crt_op = self.opt_crt.minimize(self.loss_crt, var_list=self.crt_vars)
            # grads_and_vars = self.opt_crt.compute_gradients(self.loss_crt, self.inputs.vars )

    def _build(self):
        with tf.compat.v1.variable_scope('encoder'):
            _LAYER_UIDS['encoder'] = 0
            if self.flags.num_layer == 1:
                self.signal_layers.append(GraphConvolution(input_dim=self.input_dim,
                                                           output_dim=self.flags.hidden1,
                                                           placeholders=self.placeholders,
                                                           # act=tf.nn.leaky_relu,
                                                           act=lambda x: x,
                                                           dropout=True,
                                                           sparse_inputs=True,
                                                           # bias=True,
                                                           logging=self.logging))
            else:
                self.signal_layers.append(GraphConvolution(input_dim=self.input_dim,
                                                          output_dim=self.flags.hidden1,
                                                          placeholders=self.placeholders,
                                                          act=tf.nn.leaky_relu,
                                                          dropout=True,
                                                          sparse_inputs=True,
                                                          logging=self.logging))
                for i in range(self.flags.num_layer - 2):
                    self.signal_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                              output_dim=self.flags.hidden1,
                                                              placeholders=self.placeholders,
                                                              act=tf.nn.leaky_relu,
                                                              dropout=True,
                                                              logging=self.logging))

                self.signal_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                          output_dim=self.flags.hidden1,
                                                          placeholders=self.placeholders,
                                                          # act=tf.nn.leaky_relu,
                                                           act=lambda x: x,
                                                           dropout=True,
                                                          logging=self.logging))

        with tf.compat.v1.variable_scope('actor'):
            self.actor_layers.append(Dense(input_dim=self.flags.hidden1,
                                            output_dim=self.flags.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))
            self.actor_layers.append(Dense(input_dim=self.flags.hidden1,
                                            output_dim=self.flags.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))
            self.actor_layers.append(Dense(input_dim=self.flags.hidden1,
                                            output_dim=self.flags.diver_num,
                                            placeholders=self.placeholders,
                                            # act=tf.nn.relu,
                                            act=lambda x: x,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))
        with tf.compat.v1.variable_scope('critic'):
            self.critic_layers.append(Dense(input_dim=self.flags.hidden1 + self.flags.diver_num,
                                            output_dim=self.flags.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))
            self.critic_layers.append(Dense(input_dim=self.flags.hidden1,
                                            output_dim=self.flags.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))
            self.critic_layers.append(Dense(input_dim=self.flags.hidden1,
                                            output_dim=2,
                                            placeholders=self.placeholders,
                                            # act=tf.nn.leaky_relu,
                                            act=lambda x: x,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GRNN_A2C(Model):
    '''
    GCN-based Constrained Reinforcement Learning
    '''
    def __init__(self, placeholders, flags, **kwargs):
        super(GRNN_A2C, self).__init__(**kwargs)

        self.flags = flags
        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.batchsize = placeholders['features'].get_shape().as_list()[0]
        self.inputs_action = placeholders["actions"]
        self.hidden_inputs = placeholders["hidden"]
        # self.inputs_action = tf.reshape(placeholders["actions"], (-1, self.output_dim))
        self.placeholders = placeholders

        if self.flags.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            learning_rate = tf.compat.v1.train.exponential_decay(self.flags.learning_rate,
                                                                 self.global_step_tensor,
                                                                 5000,
                                                                 self.flags.learning_decay, staircase=True)
        else:
            learning_rate = self.flags.learning_rate
        self.opt_act = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.opt_crt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.signal_layers = []
        self.state_layers = []
        self.signals = []
        self.states = []

        self.build()

    def _loss(self):
        # Weight decay loss
        l2loss_act = 0
        for var in self.actor_layers[0].vars.values():
            l2loss_act += self.flags.weight_decay * tf.nn.l2_loss(var)
        self.loss_act += l2loss_act
        l2loss_crt = 0
        for var in self.critic_layers[0].vars.values():
            l2loss_crt += self.flags.weight_decay * tf.nn.l2_loss(var)
        self.loss_crt += l2loss_crt

        # regression loss
        loss_crt = tf.sqrt(tf.reduce_mean((self.placeholders['labels'] - self.critic_pred)**2))
        # mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.critic_pred)
        # loss_mse = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # loss_mse = tf.sqrt(tf.reduce_mean((self.placeholders['labels']-self.critic_pred)**2))
        # loss_crt = tf.sqrt(tf.reduce_mean((self.placeholders['network_q']-self.critic_q)**2))
        loss_act = -tf.reduce_sum(self.critic_pred) # / (tf.cast(tf.math.count_nonzero(self.placeholders['labels']), tf.float32)+1e-6)
        # loss_act = -tf.reduce_mean(self.critic_q)
        # loss_act = tf.sqrt(tf.reduce_mean((self.placeholders['labels']-self.actor_pred)**2))
        self.act_vars = tf.compat.v1.trainable_variables(scope=self.name+'/'+'actor')
        self.crt_vars = tf.compat.v1.trainable_variables(scope=self.name+'/'+'critic')
        self.loss_act += loss_act
        self.loss_crt += loss_crt

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # signal path
        self.signals.append(self.inputs)
        for ilayer in range(0, len(self.signal_layers)):
            layer = self.signal_layers[ilayer]
            hidden = layer(self.signals[-1])
            self.signals.append(hidden)
        signal_out = self.signals[-1]
        self.states.append(self.hidden_inputs)
        for ilayer in range(0, len(self.state_layers)):
            layer = self.state_layers[ilayer]
            hidden = layer(self.states[-1])
            self.states.append(hidden)
        self.hidden_out = self.states[-1] + signal_out

        self.actors.append(self.hidden_out)
        for ilayer in range(0, len(self.actor_layers)):
            layer = self.actor_layers[ilayer]
            hidden = layer(self.actors[-1])
            self.actors.append(hidden)
        self.actor_pred = self.actors[-1]
        selector = self.placeholders['labels_mask'] * tf.ones_like(self.placeholders['labels'], dtype=tf.int32)
        selector = tf.cast(selector, tf.bool)
        self.critic_actions = tf.where(selector, self.actor_pred, self.inputs_action)
        critic_inputs = tf.concat([self.hidden_out, self.critic_actions], axis=-1)
        self.critics.append(critic_inputs)
        for ilayer in range(0, len(self.critic_layers)):
            layer = self.critic_layers[ilayer]
            hidden = layer(self.critics[-1])
            self.critics.append(hidden)
        self.critic_pred = self.critics[-1][:, 0]
        self.network_weight = tf.nn.softmax(self.critics[-1][:, 1], axis=0)
        self.critic_q = tf.reduce_sum(self.critic_pred * self.network_weight)
        # self.critic_q = self.critic_pred

        self.outputs = self.actor_pred
        self.actions = self.actor_pred
        self.qvalues = self.critic_pred
        self.outputs_softmax = self.actions # just to be compatable

    def _opt_set(self):
        if self.flags.learning_decay < 1.0:
            self.opt_act_op = self.opt_act.minimize(self.loss_act, var_list=self.act_vars, global_step=self.global_step_tensor)
            self.opt_crt_op = self.opt_crt.minimize(self.loss_crt, var_list=self.crt_vars, global_step=self.global_step_tensor)
        else:
            # gvs = self.opt_act.compute_gradients(self.loss_act, var_list=self.act_vars)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # self.opt_act_op = self.opt_act.apply_gradients(capped_gvs)
            # gvs = self.opt_crt.compute_gradients(self.loss_crt, var_list=self.crt_vars)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # self.opt_crt_op = self.opt_crt.apply_gradients(capped_gvs)
            self.opt_act_op = self.opt_act.minimize(self.loss_act, var_list=self.act_vars)
            self.opt_crt_op = self.opt_crt.minimize(self.loss_crt, var_list=self.crt_vars)
            # grads_and_vars = self.opt_crt.compute_gradients(self.loss_crt, self.inputs.vars )

    def _build(self):
        with tf.compat.v1.variable_scope('actor'):
            _LAYER_UIDS['actor'] = 0
            if self.flags.num_layer == 1:
                self.signal_layers.append(GraphConvolution(input_dim=self.input_dim,
                                                        output_dim=self.flags.hidden1,
                                                        placeholders=self.placeholders,
                                                        act=tf.nn.relu,
                                                        dropout=True,
                                                        sparse_inputs=True,
                                                        # bias=True,
                                                        logging=self.logging))
            else:
                self.signal_layers.append(GraphConvolution(input_dim=self.input_dim,
                                                          output_dim=self.flags.hidden1,
                                                          placeholders=self.placeholders,
                                                          act=tf.nn.leaky_relu,
                                                          dropout=True,
                                                          sparse_inputs=True,
                                                          logging=self.logging))
                for i in range(self.flags.num_layer - 2):
                    self.signal_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                              output_dim=self.flags.hidden1,
                                                              placeholders=self.placeholders,
                                                              act=tf.nn.leaky_relu,
                                                              dropout=True,
                                                              logging=self.logging))

                self.signal_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                          output_dim=self.flags.hidden1,
                                                          placeholders=self.placeholders,
                                                          act=tf.nn.relu,
                                                          dropout=True,
                                                          logging=self.logging))

        with tf.compat.v1.variable_scope('actor'):
            _LAYER_UIDS['actor'] = 0
            if self.flags.num_layer == 1:
                self.state_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                           output_dim=self.flags.hidden1,
                                                           placeholders=self.placeholders,
                                                           act=tf.nn.leaky_relu,
                                                           dropout=True,
                                                           bias=True,
                                                           logging=self.logging))
            else:
                self.state_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                           output_dim=self.flags.hidden1,
                                                           placeholders=self.placeholders,
                                                           act=tf.nn.leaky_relu,
                                                           dropout=True,
                                                           bias=True,
                                                           logging=self.logging))
                for i in range(self.flags.num_layer - 2):
                    self.state_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                               output_dim=self.flags.hidden1,
                                                               placeholders=self.placeholders,
                                                               act=tf.nn.leaky_relu,
                                                               dropout=True,
                                                               bias=True,
                                                               logging=self.logging))

                self.state_layers.append(GraphConvolution(input_dim=self.flags.hidden1,
                                                           output_dim=self.flags.hidden1,
                                                           placeholders=self.placeholders,
                                                           act=tf.nn.leaky_relu,
                                                           dropout=True,
                                                           bias=True,
                                                           logging=self.logging))

        with tf.compat.v1.variable_scope('actor'):
            self.actor_layers.append(Dense(input_dim=self.flags.hidden1,
                                            output_dim=self.flags.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))
            self.actor_layers.append(Dense(input_dim=self.flags.hidden1,
                                            output_dim=self.flags.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))
            self.actor_layers.append(Dense(input_dim=self.flags.hidden1,
                                            output_dim=self.flags.diver_num,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            # act=lambda x: x,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))
        with tf.compat.v1.variable_scope('critic'):
            self.critic_layers.append(Dense(input_dim=self.flags.hidden1 + self.flags.diver_num,
                                            output_dim=self.flags.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))
            self.critic_layers.append(Dense(input_dim=self.flags.hidden1,
                                            output_dim=self.flags.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))
            self.critic_layers.append(Dense(input_dim=self.flags.hidden1,
                                            output_dim=2,
                                            placeholders=self.placeholders,
                                            # act=tf.nn.leaky_relu,
                                            act=lambda x: x,
                                            bias=True,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class FUNCAPPROX(Model):
    def __init__(self, placeholders, input_dim, order, **kwargs):
        super(FUNCAPPROX, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.order = order
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1*FLAGS.learning_rate)

        self.build()

    def _loss_ce(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += my_softmax_cross_entropy(self.outputs, self.placeholders['labels'])
    def _loss(self):
        # Weight decay loss
        vars = tf.compat.v1.trainable_variables(scope=self.name)
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                           if 'bias' not in v.name])
        self.loss += FLAGS.weight_decay * lossL2

        greedy_util_est = self.placeholders['greedy'] * self.outputs
        # util_loss = tf.abs(self.placeholders['reward'] - tf.reduce_sum(greedy_util_est))/tf.reduce_sum(self.placeholders['greedy'])
        util_loss = tf.sqrt(tf.reduce_mean((self.placeholders['labels'] - self.outputs)**2))
        # regression loss
        self.loss += util_loss
        # self.loss += tf.reduce_mean(tf.square(self.outputs-self.placeholders['labels']))

    def _accuracy(self):
        self.accuracy = my_accuracy(self.outputs, self.placeholders['labels'])

    def _f1(self):
        pass

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):
        # self.layers.append(KPoly(input_dim=self.input_dim,
        #                          rank=self.order,
        #                          sparse_inputs=True,
        #                          logging=self.logging))
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.leaky_relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        for i in range(18):
            self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                     output_dim=FLAGS.hidden1,
                                     placeholders=self.placeholders,
                                     act=tf.nn.leaky_relu,
                                     dropout=True,
                                     logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class SAGE_DQN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(SAGE_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        if 'concat' in kwargs:
            self.concat = kwargs['concat']
        else:
            self.concat = False

        if FLAGS.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                      initializer=tf.zeros_initializer)
            learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.learning_rate, self.global_step_tensor, 5000,
                                                       FLAGS.learning_decay, staircase=True)
        else:
            learning_rate = FLAGS.learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        # mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:,0:self.output_dim])
        # diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs[:,0:self.output_dim])

        for i in range(1, FLAGS.diver_num):
            diver_loss = tf.reduce_min([diver_loss,
                                        tf.reduce_mean(
                                            tf.abs(self.outputs[:, i:i + self.output_dim] - self.placeholders['labels']))])
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        if not self.skip:
            self.outputs = self.activations[-1]
        else:
            # hiddens = [dense_input] + self.activations[1:]
            hiddens = [self.dense_input] + self.activations[-1:]
            super_hidden = tf.concat(hiddens, axis=1)
            if FLAGS.wts_init == 'random':
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1])
            elif FLAGS.wts_init == 'zeros':
                input_dim = self.dense_input.get_shape().as_list()[1]
                output_dim = self.activations[-1].shape.as_list()[1]
                dense_shape = [super_hidden.get_shape().as_list()[1], output_dim]
                init_wts = np.zeros(dense_shape, dtype=np.float32)
                diag_mtx = np.identity(int(output_dim/2))
                neg_indices = list(range(0, output_dim-1, 2))
                pos_indices = list(range(1, output_dim, 2))
                init_wts[0:int(output_dim/2), neg_indices] = - diag_mtx
                init_wts[0:int(output_dim/2), pos_indices] = diag_mtx
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1], kernel_initializer=tf.constant_initializer(init_wts))

        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphsage'] = 0
        i_dims = [self.input_dim]
        for i in range(1, FLAGS.num_layer):
            if self.concat:
                i_dims.append(FLAGS.hidden1 * 2)
            else: # even layer
                i_dims.append(FLAGS.hidden1)
        if self.concat:
            i_dims.append(FLAGS.hidden1 * 2)
        else:
            i_dims.append(FLAGS.hidden1)
        for i in range(0, FLAGS.num_layer):
            self.layers.append(MaxPoolingAggregator(input_dim=i_dims[i],
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            # act=lambda x: x,
                                            dropout=True,
                                            sparse_inputs=(i==0),
                                            concat=self.concat,
                                            # bias=True,
                                            logging=self.logging))

        self.layers.append(Dense(input_dim=i_dims[-1],
                                 output_dim=FLAGS.diver_num,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))


    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs

