import tensorflow as tf
from .rbm import RBM
from .util import sample_bernoulli, sample_gaussian


class GBRBM(RBM):
    def __init__(self, n_visible, n_hidden, sample_visible=False, sigma=1, lamda=0.1, sparsity=0.99, **kwargs):
        self.sample_visible = sample_visible
        self.sigma = sigma
        RBM.__init__(self, n_visible, n_hidden, lamda=lamda, sparsity=sparsity,
                     **kwargs)

    def _initialize_vars(self):
        hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        visible_recon_p = tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.visible_bias

        if self.sample_visible:
            visible_recon_p = sample_gaussian(visible_recon_p, self.sigma)

        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(self.x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        # sparsity
        lamda_ = self.lamda / (2 * hidden_p.get_shape()[0])
        # sparsity_grad_base = self.lamda * (self.rho - tf.reduce_mean(hidden_p)) * hidden_p * (1. - hidden_p)
        sparsity_grad_base = (self.rho - tf.reduce_mean(hidden_p, 0)) * hidden_p * (1. - hidden_p)
        sparsity_grad_w = lamda_ * tf.matmul(tf.transpose(self.x), sparsity_grad_base)
        sparsity_grad_hidden_bias = lamda_ * sparsity_grad_base

        def f(x_old, x_new):
            return self.momentum * x_old +\
                   self.learning_rate * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])

        delta_w_new = f(self.delta_w, positive_grad - negative_grad + sparsity_grad_w)
        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_p, 0))
        delta_hidden_bias_new = f(
            self.delta_hidden_bias,
            tf.reduce_mean(hidden_p - hidden_recon_p + sparsity_grad_hidden_bias, 0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias
        self.compute_visible_from_hidden = tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias

        # override cost calc
        self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))
        self.compute_err += self.lamda * tf.square(self.rho - tf.reduce_mean(hidden_p))
