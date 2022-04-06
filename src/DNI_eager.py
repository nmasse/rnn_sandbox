#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:03:11 2018
@author: omarschall
Modified: Matt Rosen (3/2022)
"""

import numpy as np
import tensorflow as tf
import copy
from . import util

class DNI():
    """Implements the Decoupled Neural Interface (DNI) algorithm for an RNN from
    Jaderberg et al. 2017.
    Details are in our review, original paper, or Czarnecki et al. 2017.
    Briefly, we linearly approximate the (future-facing) credit assignment
    vector c = dL/da by the variable 'sg' for 'synthetic gradient'
    (of shape (n_h)) using
    c ~ sg = A a_tilde                                                     (1)
    where a_tilde = [a; y^*; 1] is the network state concatenated with the label
    y^* and a constant 1 (for bias), of shape (m_out = n_h + n_in + 1). Then the
    gradient is calculated by combining this estimate with the immediate
    parameter influence \phi'(h) a_hat
    dL/dW_{ij} = dL/da_i da_i/dW_{ij} = sg_i alpha \phi'(h_i) a_hat_j.     (2)
    The matrix A must be updated as well to make sure Eq. (1) is a good
    estimate. Details are in our paper or the original; briefly, a bootstrapped
    credit assignment estimate is formed via
    c^* = q_prev + sg J = q_prev + (A a_tilde_prev) J                      (3)
    where J = da^t/da^{t-1} is the network Jacobian, either calculated exactly
    or approximated (see update_J_approx method). The arrays q_prev and
    a_tilde_prev are q and a_tilde from the previous time step;
    this is because the update requires reference to the "future" (by one time
    step) network state, so we simply wait a time step, such that q and a_tilde
    are now q_prev and a_tilde_prev, respectively. This target for credit
    assignment is then used to calculate a prediction loss gradient for A:
    d/dA_{ij} 0.5 * ||A a_tilde - c^*||^2 = (A a_tilde - c^*)_i a_tilde_j  (4)
    This gradient is used to update A by a given optimizer."""

    def __init__(self, rnn_params, top_down_w1, optimizer, W_rec, alpha, batch_size, 
        use_approx_J=True, W_FB=None):
        """Inits an instance of DNI by specifying the optimizer for the A
        weights and other kwargs.
        Args:
            optimizer (optimizers.Optimizer): An Optimizer instance used to
                update the weights of A based on its credit assignment
                prediction loss gradients.
        Keywords args:
            A (numpy array): Initial value of A weights, must be of shape
                (n_h, m_out). If None, A is initialized with random Gaussian.
            J_lr (float): Learning rate for learning approximate Jacobian.
            activation (functions.Function): Activation function for the
                synthetic gradient function, applied elementwise after A a_tilde
                operation. Default is identity.
            SG_label_activation (functions.Function): Activation function for
                the synthetic gradient function as used in calculating the
                *label* for the
            use_approx_J (bool): If True, trains the network using the
                approximated Jacobian rather than the exact Jacobian.
            SG_L2_reg (float): L2 regularization strength on the A weights, by
                default 0.
            fix_A_interval (int): The number of time steps to wait between
                updating the synthetic gradient method used to bootstrap the
                label estimates. Default is 5."""

        self.name = 'DNI'

        #Default parameters
        self.optimizer = optimizer
        self.SG_L2_reg = 0
        self.fix_A_interval = 5
        self.use_approx_J = use_approx_J
        self.alpha = alpha
        self.J_lr = 0.0001

        self.rnn_params = rnn_params
        self.n_top_down = self.rnn_params.n_top_down
        self.n_top_down_hidden = self.rnn_params.n_top_down_hidden
        self.n_hidden = self.rnn_params.n_hidden
        self.n_actions = self.rnn_params.n_actions
        self.m = self.n_top_down 
        self.q = tf.zeros(self.n_hidden)
        self.batch_size = batch_size

        # Bind loss fxn 
        self.L = util.softmax_cross_entropy_
        self.dL_dz = util.softmax_cross_entropy_derivative

        # Bind feedback weight matrix and top-down weight matrix
        if W_FB is None:
            self.W_FB = tf.random.normal([self.n_actions, self.n_hidden], 
                0, 1/np.sqrt(self.n_actions))
        else:
            self.W_FB = W_FB
        self.top_down_w1 = top_down_w1


        self.W_rec = tf.tile(tf.expand_dims(W_rec, 0), [self.batch_size, 1, 1])

        self.m_out = self.n_hidden + self.n_actions + 1

        # Set up Jacobian
        self.J_approx = copy.copy(W_rec)
        
        self.i_fix = 0
        self.A = tf.Variable(tf.random.normal([self.m_out, self.n_hidden],
                0, np.sqrt(1/self.m_out)))
        self.A_ = copy.copy(self.A)


    def reset_learning(self):
        """Resets internal variables of the learning algorithm (relevant if
        simulation includes a trial structure). Default is to do nothing."""
        self.q = tf.zeros(self.n_hidden)
        self.J_approx = copy.copy(self.W_rec)

        return

    #@tf.function()
    def get_outer_grads(self, h, labels, policy):
        """Calculates the derivative of the loss with respect to the output
        parameters rnn.W_out and rnn.b_out.
        Calculates the outer gradients in the manner of a perceptron derivative
        by taking the outer product of the error with the "regressors" onto the
        output (the hidden state and constant 1).
        Returns:
            A numpy array of shape (rnn.n_out, self.n_h + 1) containing the
                concatenation (along column axis) of the derivative of the loss
                w.r.t. rnn.W_out and w.r.t. rnn.b_out.
        """
        h_ = tf.concat((h, tf.ones((h.shape[0], 1))), axis=1)
        dL_dz = self.dL_dz(labels, policy)
        return tf.tensordot(dL_dz, h_, axes=([0], [0])), dL_dz

    #@tf.function()
    def propagate_feedback_to_hidden(self, dL_dz, q):
        """Performs one step of backpropagation from the outer-layer errors to
        the hidden state.
        Calculates the immediate derivative of the loss with respect to the
        hidden state. By default, this is done by taking the network's error (dL/dz)
        and applying the chain rule, i.e. taking its matrix product with the
        random matrix of feedback weights, as in feedback alignment. 
        (See Lillicrap et al. 2016.) Updates q to the current value of dL/da."""
        self.q_prev = copy.copy(self.q)
        self.q = dL_dz @ self.W_FB
        return

    def L2_regularization(self, grads):
        """Adds L2 regularization to the gradient.
        Args:
            grads (list): List of numpy arrays representing gradients before L2
                regularization is applied.
        Returns:
            A new list of grads with L2 regularization applied."""


        #Add to each grad the corresponding weight's current value, weighted
        #by the L2_reg hyperparameter.
        grads += self.SG_L2_reg * grads

        return grads

    def maintain_sparsity(self, grads):
        """"If called, modifies gradient to make 0 any parameters that are
        already 0 (only for L2 params).
        
        Args:
            grads (list): List of numpy arrays representing gradients before L2
                regularization is applied.
        Returns:
            A new list of grads with L2 regularization applied."""
            
        #Multiply each gradient by 0 if it the corresponding weight is
        #already 0
        for i, W in enumerate(grads):
            grads[i] *= (W != 0)
        return grads

    #@tf.function()
    def __call__(self, h, h_prev, pol, pol_prev, lab, lab_prev, u, td_inputs, mask):
        """Calculates the final grads for this time step.
        Assumes the user has already called self.update_learning_vars, a
        method specific to each child class of Real_Time_Learning_Algorithm
        that updates internal learning variables, e.g. the influence matrix of
        RTRL. Then calculates the outer grads (gradients of W_out and b_out),
        updates q using propagate_feedback_to_hidden, and finally calling the
        get_rec_grads method (specific to each child class) to get the gradients
        of W_rec, W_in, and b_rec as one numpy array with shape (n_h, m). Then
        these gradients are split along the column axis into a list of 5
        gradients for W_rec, W_in, b_rec, W_out, b_out. L2 regularization is
        applied if L2_reg parameter is not None.
        Returns:
            Gradient w.r.t. top_down_w0. """

        # Obtain gradients w.r.t outputs
        outer_grads, dL_dz = self.get_outer_grads(h, lab, pol)
        
        # Update learning variables for this algorithm
        self.update_learning_vars(h, lab, h_prev, lab_prev, dL_dz, self.q, mask, u)
        self.propagate_feedback_to_hidden(dL_dz, self.q)
   
        td_hidden_grads = tf.transpose(self.get_rec_grads(h_prev, u, td_inputs))

        # Modify recurrent grads (n_hidden x n_hidden) 
        # to obtain grads w.r.t. top_down_w0
        outer_grads = tf.transpose(outer_grads)
        w_out_grads = outer_grads[:-1, :]
        b_out_grads = outer_grads[-1:,:]
        td_grads = td_hidden_grads @ tf.transpose(self.top_down_w1)

        return td_grads, w_out_grads, b_out_grads

    #@tf.function()
    def update_learning_vars(self, h, label, h_prev, label_prev, dL_dz, q, mask, u):
        """Updates the A matrix by Eqs. (3) and (4)."""

        self.update_J_approx(h, h_prev, u)

        #Compute synthetic gradient estimate of credit assignment at previous
        #time step. This is NOT used to drive learning in W but rather to drive
        #learning in A.
        self.a_tilde_prev = tf.concat([h_prev, label_prev, 
            tf.ones([self.batch_size, 1])], axis=1)
        self.sg = self.synthetic_grad(self.a_tilde_prev)

        #Compute the target, error and loss for the synthetic gradient function
        self.sg_target = self.get_sg_target(h, label, dL_dz, q)
        self.A_error   = (self.sg - self.sg_target) * tf.expand_dims(mask, 1)
        #self.A_loss    = tf.reduce_mean(0.5 * tf.math.square(self.A_error))

        #Compute gradients for A
        self.scaled_A_error = self.A_error
        self.A_grad = tf.einsum('ij,ik->kj', self.scaled_A_error, self.a_tilde_prev)

        # Update synthetic gradient parameters
        self.A_grad = tf.clip_by_norm(self.A_grad, 0.5)
        self.optimizer.apply_gradients([(self.A_grad, self.A)])


        # On interval determined by self.fix_A_interval, update A_, the values
        # used to calculate the target in Eq. (3), with the latest value of A.
        if self.i_fix == self.fix_A_interval - 1:
            self.i_fix = 0
            self.A_.assign(self.A.numpy())
        else:
            self.i_fix += 1

    #@tf.function()
    def get_sg_target(self, h, label, dL_dz, q):
        """Function for generating the target for training A. Implements Eq. (3)
        using a different set of weights A_, which are static and only
        re-assigned to A  every fix_A_interval time steps.
        Returns:
            sg_target (numpy array): Array of shape (n_out) used to get error
                signals for A in update_learning_vars.
        """

        # Get latest q value, slide current q value to q_prev.
        self.propagate_feedback_to_hidden(dL_dz, q)
        self.a_tilde = tf.concat([h, label, tf.ones([self.batch_size,1])], 
            axis=1)

        # Calculate the synthetic gradient for the 'next' (really the current,
        # but next relative to the previous) time step.
        sg_next = self.synthetic_grad_(self.a_tilde)

        # Backpropagate by one time step and add to q_prev to get sg_target.
        sg_target = self.q_prev + tf.squeeze(tf.expand_dims(sg_next,1) 
            @ self.J_approx)

        return sg_target

    @tf.function()
    def update_J_approx(self, h, h_prev, u):
        """Updates the approximate Jacobian by SGD according to a squared-error
        loss function:
        J_loss = 0.5 * || J a_prev - a ||^2.                     (6)
        Thus the gradient for the Jacobian is
        dJ_loss/dJ_{ij} = (J a_prev - a)_i a_prev_j              (7).
        """
        if self.use_approx_J:
            self.J_error = tf.einsum('ij,ijk->ij', h_prev, self.J_approx) - h
            self.J_approx -= (self.J_lr * tf.einsum('ij,ik->ijk', 
                self.J_error, h_prev) / tf.reduce_max(h))
        else:
            D = tf.linalg.diag(util.relu_derivative(u)) #Nonlinearity derivative
            self.J_approx = self.alpha * tf.multiply(D, self.W_rec) + (1 - self.alpha) * tf.eye(self.n_hidden)

    def synthetic_grad(self, a_tilde):
        """Computes the synthetic gradient using current values of A.
        Retuns:
            An array of shape (n_h) representing the synthetic gradient.
        """
        self.sg_h = a_tilde @ self.A
        return self.sg_h

    def synthetic_grad_(self, a_tilde):
        """Computes the synthetic gradient using A_ and with an extra activation
        function (for DNI(b)), only for computing the label in Eq. (3).
        Retuns:
            An array of shape (n_h) representing the synthetic gradient."""
        self.sg_h_ = a_tilde @ self.A_
        return self.sg_h_ # Should this be relu(self.sg_h_)?

    def get_rec_grads(self, h_prev, u, td_inputs):
        """Computes the recurrent grads for the network by implementing Eq. (2),
        using the current synthetic gradient function.
        Note: assuming self.a_tilde already calculated by calling get_sg_target,
        which should happen by default since update_learning_vars is always
        called before get_rec_grads.
        Returns:
            An array of shape (n_h, n_top_down_hidden) representing the network 
            gradient for the recurrent parameters.
        """

        # Calculate synthetic gradient
        self.sg = self.synthetic_grad(self.a_tilde)

        # Combine the first 3 factors of the RHS of Eq. (2) into sg_scaled
        D = util.relu_derivative(u)
        self.sg_scaled = self.sg * self.alpha * D

        return tf.tensordot(self.sg_scaled, td_inputs, axes=([0], [0]))

