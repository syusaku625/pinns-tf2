from typing import List, Dict

import numpy as np
import tensorflow as tf
import h5py

class FCN(object):
    def __init__(self, layers, lb, ub, output_names, discrete: bool = False, dtype='float32') -> None:
        """Initialize a `FCN` module.

        :param layers: The list indicating number of neurons in each layer.
        :param lb: Lower bound for the inputs.
        :param ub: Upper bound for the inputs.
        :param output_names: Names of outputs of net.
        :param discrete: If the problem is discrete or not.
        """
        super().__init__()
        
        if dtype == 'float32':
            self.dtype = tf.float32
        else:
            self.dtype = None

        self.lb = tf.constant(lb, tf.float32)
        self.ub = tf.constant(ub, tf.float32)
        
        self.model = self.initalize_net(layers)
        self.output_names = output_names
        self.discrete = discrete
    
    def create_layer(self, input, num_output, initializer, activation=None, dtype=None):
        if activation is None:
            return tf.keras.layers.Dense(
                        num_output,
                        kernel_initializer=initializer,
                        dtype=tf.float32,
                    )(input)
        
        if dtype is not None:
            return tf.keras.layers.Dense(
                        num_output,
                        activation="tanh",
                        kernel_initializer=initializer,
                        dtype=dtype,
                    )(input)
        else:
            return tf.keras.layers.Dense(
                        num_output,
                        activation="tanh",
                        kernel_initializer=initializer,
                    )(input)
            
    def initalize_net(self, layers):
        initializer = tf.initializers.GlorotUniform(seed=1234)
        
        if self.dtype is None:
            inputs = tf.keras.layers.Input(shape=(layers[0],))
        else:
            inputs = tf.keras.layers.Input(shape=(layers[0],), dtype=self.dtype)
        
        z = self.create_layer(inputs, layers[1], initializer, "tanh", self.dtype)
        
        for i in range(1, len(layers) - 2):
            z = self.create_layer(z, layers[i + 1], initializer, "tanh", self.dtype)

        outputs = self.create_layer(z, layers[-1], initializer)

        neural_net = tf.keras.Model(inputs=inputs, outputs=outputs)
        neural_net.summary()
        
        return neural_net
        
    def __call__(self, spatial: List[tf.Tensor], time: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Perform a single forward pass through the network.

        :param spatial: List of input spatial tensors.
        :param time: Input tensor representing time.
        :return: A tensor of solutions.
        """

        # Discrete Mode
        if self.discrete:
            if len(spatial) == 2:
                x, y = spatial
                z = tf.concat((x, y), 1)
            elif len(spatial) == 3:
                x, y, z = spatial
                z = tf.concat((x, y, z), 1)
            else:
                z = spatial[0]
            z = 2.0 * (z - self.lb[:-1]) / (self.ub[:-1] - self.lb[:-1]) - 1.0

        # Continuous Mode
        else:
            if len(spatial) == 1:
                x = spatial[0]
                z = tf.concat((x, time), 1)
            elif len(spatial) == 2:
                x, y = spatial
                z = tf.concat((x, y, time), 1)
            else:
                x, y, z = spatial
                z = tf.concat((x, y, z, time), 1)
            z = 2.0 * (z - self.lb) / (self.ub - self.lb) - 1.0

        z = self.model(z)

        # Discrete Mode
        if self.discrete:
            outputs_dict = {name: z for i, name in enumerate(self.output_names)}

        # Continuous Mode
        else:
            outputs_dict = {name: z[:, i : i + 1] for i, name in enumerate(self.output_names)}
            
        return outputs_dict

class NetHFM(object):
    """A simple fully-connected neural net for solving equations.

    In this model, mean and std will be used for normalization of input data. Also, weight
    normalization will be done.
    """
    output_names: List[str]
    
    def __init__(self, mean, std, layers: List, output_names: List):
        super().__init__()
        """Initialize a `NetHFM` module.

        :param mesh: The number of layers.
        :param layers: The list indicating number of neurons in each layer.
        :param output_names: Names of outputs of net.
        """
        self.num_layers = len(layers)
        self.output_names = output_names
        self.trainable_variables = []

        self.X_mean = tf.constant(mean, dtype=tf.float32)
        self.X_std = tf.constant(std, dtype=tf.float32)

        self.initalize_net(layers)

    def read_h5_trained_data(self, filename):
        loop = 13
        with h5py.File(filename, 'r') as f:
            weights, biases, gammas = [], [], []
            for i in range (0, loop):
                tmp_weight = f['weights'][str(i)]
                weights.append(tmp_weight[()])
            for i in range (0, loop):
                tmp_biase = f['biases'][str(i)]
                biases.append(tmp_biase[()])
            for i in range (0, loop):
                tmp_gamma = f['gammas'][str(i)]
                gammas.append(tmp_gamma[()])
        
        return weights, biases, gammas
    
        
    def initalize_net(self, layers: List) -> None:
        """Initialize the neural network weights, biases, and gammas.

        :param layers: The list indicating number of neurons in each layer.
        """
        self.weights = []
        self.biases = []
        self.gammas = []
        #self.W = tf.Variable(tf.random.uniform([4, 20 //2  ], minval=-0.04, maxval=0.04, dtype=tf.float32) ,dtype=tf.float32, trainable=False)
        #self.weights.append(self.W)
        #if use trained data comment out!
        filename = "199999_trained.h5"
        read_weights, read_biases, read_gammas = self.read_h5_trained_data(filename)
        for i in range(0, 13):
            self.weights.append(tf.Variable(read_weights[i], dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(read_gammas[i], dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(read_biases[i], dtype=tf.float32, trainable=True))
        
        #if not use trained data comment out!
        #W1 = np.random.normal(size=[4, layers[1]])
        #W2 = np.random.normal(size=[4, layers[1]])
        #self.weights.append(tf.Variable(W1, dtype=tf.float32, trainable=True))
        #self.weights.append(tf.Variable(W2, dtype=tf.float32, trainable=True))
        #b1 = np.zeros([1, layers[1]])
        #b2 = np.zeros([1, layers[1]])
        #self.biases.append(tf.Variable(b1, dtype=tf.float32, trainable=True))
        #self.biases.append(tf.Variable(b2, dtype=tf.float32, trainable=True))
        #g1 = np.ones([1, layers[1]])
        #g2 = np.ones([1, layers[1]])
        #self.gammas.append(tf.Variable(g1, dtype=tf.float32, trainable=True))
        #self.gammas.append(tf.Variable(g2, dtype=tf.float32, trainable=True))
    #
        #for l in range(0,self.num_layers-1):
        #    in_dim = layers[l]
        #    out_dim = layers[l+1]
        #    W = np.random.normal(size=[in_dim, out_dim])
        #    b = np.zeros([1, out_dim])
        #    g = np.ones([1, out_dim])
        #    # tensorflow variables
        #    self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
        #    self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
        #    self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))
#
        self.trainable_variables.extend(self.weights)
        self.trainable_variables.extend(self.biases)
        self.trainable_variables.extend(self.gammas)

    
    def __call__(self, spatial: List[tf.Tensor], time: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Perform a forward pass through the network.

        :param spatial: List of input spatial tensors.
        :param time: Input tensor representing time.
        :return: A dictionary with output names as keys and corresponding output tensors as values.
        """
        if len(spatial) == 1:
            x = spatial[0]
            H = tf.concat((x, time), 1)
        elif len(spatial) == 2:
            x, y = spatial
            H = tf.concat((x, y, time), 1)
        else:
            x, y, z = spatial
            H = tf.concat((x, y, z, time), 1)
        H = (H - self.X_mean) / self.X_std
        W1 = self.weights[0]
        W2 = self.weights[1]
        b1 = self.biases[0]
        b2 = self.biases[1]
        g1 = self.gammas[0]
        g2 = self.gammas[1]
        V1 = W1/tf.norm(W1, axis = 0, keepdims=True)
        V2 = W2/tf.norm(W2, axis = 0, keepdims=True)
        H1 = tf.matmul(H, V1)
        H2 = tf.matmul(H, V2)
        H1 = g1 * H1 + b1
        H2 = g2 * H2 + b2
        #U_i = H1 * tf.sigmoid(H1)
        #V_i = H2 * tf.sigmoid(H2)
        U_i = tf.tanh(H1)
        V_i = tf.tanh(H2)
        #H = (tf.concat([tf.sin(tf.matmul(H, self.weights[0])), tf.cos(tf.matmul(H, self.weights[0]))], 1))
        for i, (W, b, g) in enumerate(zip(self.weights, self.biases, self.gammas)):
            if i == 0 or i==1:
                continue
            # weight normalization
            V = self.weights[i]/tf.norm(self.weights[i], axis = 0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g * H + b
            # activation
            if i < self.num_layers-3:
                H = tf.add(tf.math.multiply(H, U_i), tf.math.multiply(1.0 - H, V_i))
                #H = tf.tanh(H)
                H = H * tf.sigmoid(H)
            else:
                H = tf.tanh(H)
                #H = H * tf.sigmoid(H)
        outputs_dict = {name: H[:, i : i + 1] for i, name in enumerate(self.output_names)}

        return outputs_dict


if __name__ == "__main__":
    _ = FCN()
    _ = NetHFM()