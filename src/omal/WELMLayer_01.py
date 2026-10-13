import h5py

from Layers.ELMLayer import ELMLayer
from Optimizers.ELMOptimizer import ELMOptimizer
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf
from Resources.gram_schmidt import gram_schmidt


class WELMLayer(ELMLayer):
    """
        Weighted Extreme Learning Machine Layer.

        This layer extends the functionality of the ELMLayer by incorporating weighted samples during training.

        Args:
        -----------
            number_neurons (int): The number of neurons in the hidden layer.
            activation (str, optional): The activation function to use. Defaults to 'tanh'.
            act_params (dict, optional): Additional parameters for the activation function. Defaults to None.
            C (float, optional): Regularization parameter. Defaults to 1.0.
            beta_optimizer (ELMOptimizer, optional): Optimizer for updating output weights. Defaults to None.
            is_orthogonalized (bool, optional): Whether to orthogonalize the hidden layer weights. Defaults to False.
            weight_method (str, optional): Method for computing sample weights. Defaults to 'wei-1'. Oprions: 'wei-1',
            'wei-2', 'ban-1', 'ban-decay'
            weight_param (int, optional): Parameter for weight computation (if applicable). Defaults to 4.
            **params: Additional parameters.

        Attributes:
        -----------
            weight_method (str): Method for computing sample weights.
            weight_param (int): Parameter for weight computation (if applicable).

        Methods:
        -----------
            fit(x, y): Fit the layer to the input-output pairs.
            predict(x): Predict the output for input data.
            predict_proba(x): Predict class probabilities for input data.
            calc_output(x): Calculate the output for input data.
            count_params(): Count the number of trainable and non-trainable parameters.
            to_dict(): Convert layer attributes to a dictionary.
            load(attributes): Load layer attributes from a dictionary.

        Example:
        -----------
        >>> layer = WELMLayer(number_neurons=1000, activation='tanh', weight_method='wei-1')
        >>> model = ELMModel(layer)
    """
    def __init__(self, number_neurons, activation='tanh', act_params=None, C=1.0, beta_optimizer: ELMOptimizer = None,
                 is_orthogonalized=False, weight_method='wei-1', weight_param=4, **params):
        super().__init__(number_neurons, activation, act_params, C, beta_optimizer, is_orthogonalized, **params)
        self.weight_method = weight_method
        self.weight_param = weight_param

    def fit(self, x, y):
        """
            Fit the WELMLayer to the input-output pairs.

            Args:
            -----------
                x (tf.Tensor): Input data.
                y (tf.Tensor): Output data.

            Returns:
            -----------
                None

            Example:
            -----------
            >>> layer = WELMLayer(number_neurons=1000, activation='tanh', weight_method='wei-1')
            >>> layer.build(X.shape)
            >>> layer.fit(X_train, y_train)

        """
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        self.input = x
        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)

                # Compute sample weights according to the WELM formulation
        # used in OMAL.
        #
        # For binary classification:
        #   positive samples -> 1 / N_positive
        #   negative samples -> 1 / N_negative

        if self.weight_method == 'wei-1':

            n_positive = tf.reduce_sum(y)
            n_negative = tf.cast(tf.shape(y)[0], tf.float32) - n_positive

            eps = tf.constant(1e-12, dtype=tf.float32)

            positive_weight = 1.0 / tf.maximum(n_positive, eps)
            negative_weight = 1.0 / tf.maximum(n_negative, eps)

            # Each sample receives the weight corresponding
            # to its binary class.
            w = (
                y * positive_weight
                + (1.0 - y) * negative_weight
            )

        else:
            raise ValueError(
                f"Weight method '{self.weight_method}' "
                "is not implemented for the OMAL WELM."
            )

        # w has shape (N, 1)
        # No explicit N x N diagonal matrix W is constructed.

                # Apply sample weights without constructing the
        # explicit diagonal W matrix.
        WH = w * H
        Wy = w * y

        if x.shape[0] < self.number_neurons:

            # Dual formulation:
            # beta = H^T (W H H^T + I/C)^(-1) W y

            Hp = tf.matmul(WH, H, transpose_b=True)

            if self.C is not None:
                regularization = 1.0 / self.C

                Hp = tf.linalg.set_diag(
                    Hp,
                    tf.linalg.diag_part(Hp) + regularization
                )

            # Solve:
            # (W H H^T + I/C) Z = W y
            Z = tf.linalg.solve(Hp, Wy)

            # beta = H^T Z
            beta = tf.matmul(H, Z, transpose_a=True)

        else:

            # Primal formulation:
            # beta = (H^T W H + I/C)^(-1) H^T W y

            Hp = tf.matmul(H, WH, transpose_a=True)

            if self.C is not None:
                regularization = 1.0 / self.C
                Hp = tf.linalg.set_diag(
                    Hp,
                    tf.linalg.diag_part(Hp) + regularization
                )

            rhs = tf.matmul(H, Wy, transpose_a=True)

            beta = tf.linalg.solve(Hp, rhs)

        if self.beta_optimizer is not None:
            self.beta, self.error_history = self.beta_optimizer.optimize(beta, H, y)
        else:
            self.beta = beta

        self.feature_map = H
        self.output = tf.matmul(H, self.beta)

    def predict(self, x):
        """
            Predict the output for input data.

            Args:
            -----------
                x (tf.Tensor): Input data.

            Returns:
            -----------
                tf.Tensor: Predicted output.

            Example:
            -----------
            >>> layer = WELMLayer(number_neurons=1000, activation='tanh', weight_method='wei-1')
            >>> layer.build(X.shape)
            >>> layer.fit(X_train, y_train)
            >>> pred = layer.predict(X_test)
        """
        x = tf.cast(x, dtype=tf.float32)
        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)
        output = tf.matmul(H, self.beta)
        return output

    def predict_proba(self, x):
        """
            Predict class probabilities for input data.

            Args:
            -----------
                x (tf.Tensor): Input data.

            Returns:
            -----------
                tf.Tensor: Predicted class probabilities.

            Example:
            -----------
            >>> layer = WELMLayer(number_neurons=1000, activation='tanh', weight_method='wei-1')
            >>> layer.build(X.shape)
            >>> layer.fit(X_train, y_train)
            >>> pred = layer.predict_proba(X_test)
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)

        return tf.keras.activations.sigmoid(pred)

    def calc_output(self, x):
        """
            Calculate the output for input data.

            Args:
            -----------
                x (tf.Tensor): Input data.

            Returns:
            -----------
                tf.Tensor: Calculated output.
        """
        x = tf.cast(x, dtype=tf.float32)
        out = self.activation(tf.matmul(x, self.beta, transpose_b=True))
        self.output = out
        return out

    def __str__(self):
        """
            String representation of the layer.

            Returns:
            -----------
                str: String representation.
        """
        return f"{self.name}, neurons: {self.number_neurons}"

    def count_params(self):
        """
            Count the number of trainable and non-trainable parameters.

            Returns:
            -----------
                dict: Dictionary containing counts of trainable and non-trainable parameters.
        """
        if self.beta is None:
            trainable = 0
        else:
            trainable = self.beta.shape[0] * self.beta.shape[1]
        if self.alpha is None or self.bias is None:
            non_trainable = 0
        else:
            non_trainable = self.alpha.shape[0] * self.alpha.shape[1] + self.bias.shape[0]
        return {'trainable': trainable, 'non_trainable': non_trainable, 'all': trainable + non_trainable}

    def to_dict(self):
        """
            Convert layer attributes to a dictionary.

            Returns:
            -----------
                dict: Dictionary containing layer attributes.
        """
        attributes = {
            'name': 'WELMLayer',
            'number_neurons': self.number_neurons,
            'activation': self.activation_name,
            'act_params': self.act_params,
            'C': self.C,
            'is_orthogonalized': self.is_orthogonalized,
            "beta": self.beta,
            "alpha": self.alpha,
            "bias": self.bias,
            "weight_method": self.weight_method,
            "weight_param": self.weight_param
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
            Load layer attributes from a dictionary.

            Args:
            -----------
                attributes (dict): Dictionary containing layer attributes.

            Returns:
            -----------
                WELMLayer: Loaded WELMLayer instance.
        """
        return cls(**attributes)
