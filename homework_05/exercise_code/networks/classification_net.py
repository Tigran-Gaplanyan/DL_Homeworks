import numpy as np
import os
import pickle

from exercise_code.networks.layer import affine_forward, affine_backward, Sigmoid, Tanh, LeakyRelu, Relu
from exercise_code.networks.base_networks import Network


class ClassificationNet(Network):
    """
    A fully-connected classification neural network with configurable 
    activation function, number of layers, number of classes, hidden size and
    regularization strength. 
    """

    def __init__(self, activation=Sigmoid(), num_layer=2,
                 input_size=3 * 32 * 32, hidden_size=100,
                 std=1e-3, num_classes=10, reg=0, **kwargs):
        """
        :param activation: choice of activation function. It should implement
            a forward() and a backward() method.
        :param num_layer: integer, number of layers. 
        :param input_size: integer, the dimension D of the input data.
        :param hidden_size: integer, the number of neurons H in the hidden layer.
        :param std: float, standard deviation used for weight initialization.
        :param num_classes: integer, number of classes.
        :param reg: float, regularization strength.
        """
        super(ClassificationNet, self).__init__("cifar10_classification_net")

        self.activation = activation
        self.reg_strength = reg

        self.cache = None

        self.memory = 0
        self.memory_forward = 0
        self.memory_backward = 0
        self.num_operation = 0

        # Initialize random gaussian weights for all layers and zero bias
        self.num_layer = num_layer
        self.params = {'W1': std * np.random.randn(input_size, hidden_size),
                       'b1': np.zeros(hidden_size)}

        for i in range(num_layer - 2):
            self.params['W' + str(i + 2)] = std * np.random.randn(hidden_size,
                                                                  hidden_size)
            self.params['b' + str(i + 2)] = np.zeros(hidden_size)

        self.params['W' + str(num_layer)] = std * np.random.randn(hidden_size,
                                                                  num_classes)
        self.params['b' + str(num_layer)] = np.zeros(num_classes)

        self.grads = {}
        self.reg = {}
        for i in range(num_layer):
            self.grads['W' + str(i + 1)] = 0.0
            self.grads['b' + str(i + 1)] = 0.0

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: Input data of shape N x D. Each X[i] is a training sample.
        :return: Predicted value for the data in X, shape N x 1
                 1-dimensional array of length N with the classification scores.
        """

        self.cache = {}
        self.reg = {}
        X = X.reshape(X.shape[0], -1)
        # Unpack variables from the params dictionary
        for i in range(self.num_layer - 1):
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]

            # Forward i_th layer
            X, cache_affine = affine_forward(X, W, b)
            self.cache["affine" + str(i + 1)] = cache_affine

            # Activation function
            X, cache_sigmoid = self.activation.forward(X)
            self.cache["sigmoid" + str(i + 1)] = cache_sigmoid

            # Store the reg for the current W
            self.reg['W' + str(i + 1)] = np.sum(W ** 2) * self.reg_strength

        # last layer contains no activation functions
        W, b = self.params['W' + str(self.num_layer)],\
               self.params['b' + str(self.num_layer)]
        y, cache_affine = affine_forward(X, W, b)
        self.cache["affine" + str(self.num_layer)] = cache_affine
        self.reg['W' + str(self.num_layer)] = np.sum(W ** 2) * self.reg_strength

        return y

    def backward(self, dy):
        """
        Performs the backward pass of the model.

        :param dy: N x 1 array. The gradient wrt the output of the network.
        :return: Gradients of the model output wrt the model weights
        """

        # Note that last layer has no activation
        cache_affine = self.cache['affine' + str(self.num_layer)]
        dh, dW, db = affine_backward(dy, cache_affine)
        self.grads['W' + str(self.num_layer)] = \
            dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
        self.grads['b' + str(self.num_layer)] = db

        # The rest sandwich layers
        for i in range(self.num_layer - 2, -1, -1):
            # Unpack cache
            cache_sigmoid = self.cache['sigmoid' + str(i + 1)]
            cache_affine = self.cache['affine' + str(i + 1)]

            # Activation backward
            dh = self.activation.backward(dh, cache_sigmoid)

            # Affine backward
            dh, dW, db = affine_backward(dh, cache_affine)

            # Refresh the gradients
            self.grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * \
                                           self.params['W' + str(i + 1)]
            self.grads['b' + str(i + 1)] = db

        return self.grads

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))

    def get_dataset_prediction(self, loader):
        scores = []
        labels = []
        
        for batch in loader:
            X = batch['image']
            y = batch['label']
            score = self.forward(X)
            scores.append(score)
            labels.append(y)
            
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()

        return labels, preds, acc


class MyOwnNetwork(ClassificationNet):
    """
    Your first fully owned network!
    
    You can define any arbitrary network architecture here!
    
    As a starting point, you can use the code from ClassificationNet above as 
    reference or even copy it to MyOwnNetwork, but of course you're also free 
    to come up with a complete different architecture and add any additional 
    functionality! (Without renaming class functions though ;))
    """

    def __init__(self, activation=LeakyRelu(), num_layer=4,
                 input_size=3 * 32 * 32, hidden_size=200,
                 std=1e-3, num_classes=10, reg=0, **kwargs):
        """
        Your network initialization. For reference and starting points, check
        out the classification network above.
        """

        super(MyOwnNetwork, self).__init__(activation, num_layer, input_size,
                                          hidden_size, std, num_classes, reg, **kwargs)

        ########################################################################
        # TODO:  Your initialization here                                      #
        ########################################################################
        # Same as ClassificationNet
        
        self.activation = activation
        self.reg_strength = reg

        self.cache = None

        self.memory = 0
        self.memory_forward = 0
        self.memory_backward = 0
        self.num_operation = 0
        
#         # Initialize random gaussian weights for all layers and zero bias
#         self.num_layer = num_layer
#         self.params = {'W1': std * np.random.randn(input_size, hidden_size),
#                        'b1': np.zeros(hidden_size)}

#         for i in range(num_layer - 2):
#             self.params['W' + str(i + 2)] = std * np.random.randn(hidden_size,
#                                                                   hidden_size)
#             self.params['b' + str(i + 2)] = np.zeros(hidden_size)

#         self.params['W' + str(num_layer)] = std * np.random.randn(hidden_size,
#                                                                   num_classes)
#         self.params['b' + str(num_layer)] = np.zeros(num_classes)

#         self.grads = {}
#         self.reg = {}
#         for i in range(num_layer):
#             self.grads['W' + str(i + 1)] = 0.0
#             self.grads['b' + str(i + 1)] = 0.0
            
        # Xavier/Glorot initialization of weights
        self.num_layer = num_layer
        limit = np.sqrt(6 / (input_size + num_classes))  # Xavier/Glorot initialization limit

        self.params = {'W1': np.random.uniform(-limit, limit, size=(input_size, hidden_size)),
                       'b1': np.zeros(hidden_size)}

        for i in range(num_layer - 2):
            self.params['W' + str(i + 2)] = np.random.uniform(-limit, limit, size=(hidden_size, hidden_size))
            self.params['b' + str(i + 2)] = np.zeros(hidden_size)

        self.params['W' + str(num_layer)] = np.random.uniform(-limit, limit, size=(hidden_size, num_classes))
        self.params['b' + str(num_layer)] = np.zeros(num_classes)  
        
        self.grads = {}
        self.reg = {}
        for i in range(num_layer):
            self.grads['W' + str(i + 1)] = 0.0
            self.grads['b' + str(i + 1)] = 0.0

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, X):
        out = None
        ########################################################################
        # TODO:  Your forward here                                             #
        ########################################################################
        # Same as ClassificationNet
        self.cache = {}
        self.reg = {}
        X = X.reshape(X.shape[0], -1)
        # Unpack variables from the params dictionary
        for i in range(self.num_layer - 1):
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]

            # Forward i_th layer
            X, cache_affine = affine_forward(X, W, b)
            self.cache["affine" + str(i + 1)] = cache_affine
            
            # Batch Normalization
            eps = 1e-5
            momentum = 0.9
            gamma = 1.0
            beta = 0

            N, D = X.shape
            running_mean = np.zeros(D, dtype=X.dtype)
            running_var = np.zeros(D, dtype=X.dtype)

            sample_mean = np.mean(X, axis=0)
            sample_var = np.var(X, axis=0)
            X = (X - sample_mean) / np.sqrt(sample_var + eps)
            out = gamma * X + beta

            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
            
            if i > 0:
                batch_size, _ = X.shape
                cache_batchnorm = (X, sample_mean, sample_var, eps, gamma, beta, batch_size)
                self.cache["BatchNorm" + str(i)] = cache_batchnorm
            else:
                cache_batchnorm = (X, sample_mean, sample_var, eps, gamma, beta)
                self.cache["BatchNorm" + str(i)] = cache_batchnorm

            # Activation function
            X, cache_sigmoid = self.activation.forward(X)
            self.cache["sigmoid" + str(i + 1)] = cache_sigmoid

            # Store the reg for the current W
            self.reg['W' + str(i + 1)] = np.sum(W ** 2) * self.reg_strength

        # last layer contains no activation functions
        W, b = self.params['W' + str(self.num_layer)],\
               self.params['b' + str(self.num_layer)]
        
        out, cache_affine = affine_forward(X, W, b)
        self.cache["affine" + str(self.num_layer)] = cache_affine
        self.reg['W' + str(self.num_layer)] = np.sum(W ** 2) * self.reg_strength

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return out

    def backward(self, dy):
        grads = {}
        ########################################################################
        # TODO:  Your backward here                                            #
        ########################################################################
        # Same as ClassificationNet
        
        # Note that last layer has no activation
        cache_affine = self.cache['affine' + str(self.num_layer)]
        dh, dW, db = affine_backward(dy, cache_affine)
        self.grads['W' + str(self.num_layer)] = \
            dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
        grads['b' + str(self.num_layer)] = db

        # The rest sandwich layers
        for i in range(self.num_layer - 2, -1, -1):
            # Unpack cache
            cache_sigmoid = self.cache['sigmoid' + str(i + 1)]
            cache_affine = self.cache['affine' + str(i + 1)]

            # Activation backward
            dh = self.activation.backward(dh, cache_sigmoid)
            
            # Batch Normalization backward
            if self.num_layer > 2:
                if i > 1:
                    cache_batchnorm = self.cache['BatchNorm' + str(i)]
                          
                    dgamma = np.sum(dy.T.dot(cache_batchnorm[0]), axis=0)
                    dbeta = np.sum(dy, axis=0)
                    dx = (1. / cache_batchnorm[6]) * cache_batchnorm[4] * (
                        (cache_batchnorm[6] * np.sum(dy, axis=0)) - (cache_batchnorm[0].dot(np.sum(dy.T.dot(cache_batchnorm[0]), axis=0)))
                    )      
                          
                    # Affine backward
                    dh, dW, db = affine_backward(dx, cache_affine)

                    # Refresh the gradients
                    grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * \
                                               self.params['W' + str(i + 1)]
                    grads['b' + str(i + 1)] = db 
                
            else:    

                # Affine backward
                dh, dW, db = affine_backward(dh, cache_affine)

                # Refresh the gradients
                grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * \
                                           self.params['W' + str(i + 1)]
                grads['b' + str(i + 1)] = db 

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return grads

