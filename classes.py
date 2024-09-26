import numpy as np
import pickle
import copy


### ---------------------------------------- ###
###                                          ###
###                   LAYERS                 ###
###                                          ###
### ---------------------------------------- ###

# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # W [N_I x N_N]
        self.biases = np.zeros((1, n_neurons)) # B [1 x N_N]

        # Regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        # The calculation performed is O [N_B x N_N] = I [N_B x N_I] * W [N_I x N_N] + B [1 x N_N] 
        # Note: the bias is a list or row vector. The operation results in adding the bias row to each 
        # row in the resulting product.
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        """
        Performs a backward pass through the layer. 
        Requires the partial derivatives from the consecutive layer, of the form:
        [ aZ1/aX1 aZ1/aX2 ... aZ1/aXn ]
        [ aZ2/aX1 aZ2/aX2 ... aZ2/aXn ]
        [            ...              ]
        [ aZn/aX1 aZn/aX2 ... aZn/aXn ]
        Note that summing along columns, as done in the case of the bias gradient
        calculation, is summing the partial derivatives relating all of the neurons
        from the consecutive layer and a given neuron from the current layer.
        """
        # Gradient of the neuron's output with respect to the weights [N_I x N_N] is 
        # I^T [N_I x N_B] * [next layer's backpropagated gradient vector][N_B x N_N]
        self.dweights = np.dot(self.inputs.T, dvalues)

        # Gradient of the neuron's output [1 x N_N] with respect to the biases is just the sum of 
        # derivatives respect to a particular neuron(input to the next layer), which are the columns of
        # the matrix of gradients.
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization 
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
             
        # Gradient on inputs
        self.dinputs = np.dot(dvalues, self.weights.T)

    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases
    
    # Set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


# Dropout
class Layer_Dropout():
    
    def __init__(self, rate) -> None:
        """
        Takes a rate of dropout as initialization parameter. 
        """
        self.rate = 1 - rate

    def forward(self, inputs, training):
        
        # Store inputs
        self.input = inputs

        # If not training, return values
        if not training:
            self.output = inputs.copy()
            return
        
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


# Input layer
class Layer_Input():

    def forward(self, inputs, training):
        self.output = inputs



### ---------------------------------------- ###
###                                          ###
###            ACTIVATION FUNCTIONS          ###
###                                          ###
### ---------------------------------------- ###


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        """
        Calculates the partial derivatives of the Softmax function's outputs with respect to each of its inputs:
        aS_i,j / aZ_i,k = S_i,j * (Kronecker_j,k - S_i,j * S_i,k). 
        The result of this calculation is the JACOBIAN MATRIX, of the form:
        [ S00 S01 S02 ... S0N ]
        [ S10 S11 S12 ... S1N ]
        [         ...         ]
        [ SN0 SN1 SN2 ... SNN ]
        Note that this is not the optimal way to calculate. Refer to the conjunct calculation in the class containing
        both the Softmax activation and Categorical Cross-Entropy loss.
        """

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)
            

    def predictions(self, outputs):
        return np.argmax(outputs, axis=-1)


# Sigmoid Activation Function
class Activation_Sigmoid():

    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1


# Linear activation
class Activation_Linear():

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs



### ---------------------------------------- ###
###                                          ###
###                OPTIMIZERS                ###
###                                          ###
### ---------------------------------------- ###



# SGD optimizer
class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates


    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# Adagrad optimizer
class Optimizer_Adagrad:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# RMSprop optimizer
class Optimizer_RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# Adam optimizer
class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)
        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1



### ---------------------------------------- ###
###                                          ###
###               LOSS CLASSES               ###
###                                          ###
### ---------------------------------------- ###



# Common loss class
class Loss:

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, *, include_regularization=False):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        # Return loss
        return data_loss, self.regularization_loss()
    
    def new_pass(self):

        # Reset variables for accumulated loss
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    def calculate_accumulated(self, *, include_regularization=False):

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss, return it
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    def regularization_loss(self):

        # 0 by default
        regularization_loss = 0 

        for layer in self.trainable_layers:
            # L1 regularization weights. Only calculate when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            
            # L1 regularizations - biases. Only calculate when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Binary cross-entropy loss
class Loss_BinaryCrossEntropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by zero, without 
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # Number of outputs per sample
        outputs = len(dvalues[0])

        # Clip
        clipped_values = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate Gradient
        self.dinputs = -(y_true / clipped_values - (1 - y_true) / (1-clipped_values)) / outputs

        # Normalize
        self.dinputs = self.dinputs / samples


# MSE Loss
class Loss_MeanSquaredError(Loss):

    # Forward pass 
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Gradient
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize
        self.dinputs = self.dinputs / samples


# Mean Absolute Error
class Loss_MeanAbsoluteError(Loss):
    
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean(np.abs(y_true-y_pred), axis=-1)

        return sample_losses
    
    def backward(self, dvalues, y_true):
        
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples



### ---------------------------------------- ###
###                                          ###
###              MODEL CLASSES               ###
###                                          ###
### ---------------------------------------- ###



class Model():
    
    # Initialize
    def __init__(self) -> None:
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def train(self, x, y, *, epochs=1, batch_size=None, 
              print_every=1, validation_data=None):

        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not being set
        train_steps = 1

        # If there is validation data passed, set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1

            # For readability
            x_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(x) // batch_size

            # If round down leaves data out
            if train_steps * batch_size < len(x):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(x_val) // batch_size

                if validation_steps * batch_size < len(x_val):
                    validation_steps += 1

        # Main training loop
        for epoch in range(1, epochs+1):

            print(f'epoch: {epoch}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):

                # If batch size not set, train using one step and full dataset
                if batch_size is None:
                    batch_x = x
                    batch_y = y
                    # Otherwise, slice a batch
                else:
                    batch_x = x[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Forward pass
                output = self.forward(batch_x, training=True)

                # Loss calculations
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
            
                # Get measures of current progress
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                
                # Backward pass
                self.backward(output, batch_y)

                # Optimize 
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Provide some insight
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                        f'accuracy:{accuracy:.3f}, ' +
                        f'loss: {loss:.3f}, ' +
                        f'data_loss: {data_loss:.3f}, ' +
                        f'reg_loss: {regularization_loss:.3f}, ' +
                        f'lr: {self.optimizer.current_learning_rate}')
                    
            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' + 
                f'accuracy:{epoch_accuracy:.3f}, ' +
                f'loss: {epoch_loss:.3f}, ' +
                f'data_loss: {epoch_data_loss:.3f}, ' +
                f'reg_loss: {epoch_regularization_loss:.3f}, ' +
                f'lr: {self.optimizer.current_learning_rate}')

            # Validation step     
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)

    def finalize(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If first layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # If middle layers
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # If last layer
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output
    
    def backward(self, output, y):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # Call method on combined activation/loss
            self.softmax_classifier_output.backward(output, y)

            # Set dinputs for last layer
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Backward pass loop
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return
        
        # Otherwise call backward method on the loss to set dinputs
        self.loss.backward(output,y)

        # Call backward method going through all the objects in reversed order passing dinputs as parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    def evaluate(self, x_val, y_val, *, batch_size=None):

        # Default value if batch size not set
        validation_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(x_val) // batch_size
            if validation_steps * batch_size < len(x_val):
                validation_steps += 1
        
        # Reset accumulated values
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_x = x_val
                batch_y = y_val
            else:
                batch_x = x_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]
        
            # Forward pass
            output = self.forward(batch_x, training=False)

            # Loss
            self.loss.calculate(output, batch_y)

            # Predictions and accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print summary
        print(f'Validation -->  ' +
            f'accuracy:{validation_accuracy:.3f}, ' +
            f'loss: {validation_loss:.3f}, '
            )

    def get_parameters(self):

        # Create a list for parameters
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters

    def set_parameters(self, parameters):

        # Iterate over the parameters and layers to update
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):

        # Open a file in binary-write mode and save parameters to it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):

        # Open a file in binary-read mode, load weights and update trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):

        # Make a deep copy of the current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from input layer and gradients from loss
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):

        # Open file in binary-read mode, load
        with open(path, 'rb') as f:
            model = pickle.load(f)

        # Return the model
        return model
    
    def predict(self, x, *, batch_size=None):
        
        # Default value if no batch size
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(x) // batch_size
            if prediction_steps * batch_size < len(x):
                prediction_steps += 1
        
        # Model's outputs
        output = []

        for step in range(prediction_steps):
            if batch_size is None:
                batch_x = x

            else:
                batch_x = x[step*batch_size:(step+1)*batch_size]

            batch_output = self.forward(batch_x, training=False)

            output.append(batch_output)

        return np.vstack(output)



### ---------------------------------------- ###
###                                          ###
###             ACCURACY CLASSES             ###
###                                          ###
### ---------------------------------------- ###

class Accuracy():
    
    def calculate(self, predictions, y):

        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy
    
    def calculate_accumulated(self):

        # Calculate accumulated
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy
    
    def new_pass(self):

        # Reset variables
        self.accumulated_sum = 0
        self.accumulated_count = 0
    

class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
    

class Accuracy_Categorical(Accuracy):

    # No init needed
    def init(self, y):
        pass

    # Compares predictions to ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
    
