#!/usr/bin/env python
"""
    Experiment with an 'auto-encoder' type architecture to generate a speech
    mask...i.e., a mask with ones at time/frequency points that look like speech
    but zero for other sounds.
"""

from audio_utils import SoundClip
from pickle import load
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, DropoutWrapper, GRUCell


class SpeechEncoderModel(object):
    """This encapsulates the neural network, including the loss function.

    """

    def __init__(self, n_steps, n_inputs, n_hidden, keep_prob=1.0):

        # construct the rnn model
        n_outputs = n_inputs

        learning_rate = 0.001

        self.X = tf.placeholder(tf.float32,
                                [None, n_steps, n_inputs], name='inputs')

        with tf.variable_scope('rnn'):
            hidden_cell = DropoutWrapper(BasicRNNCell(  # GRUCell, BasicRNNCell
                                                 num_units=n_hidden,
                                                 activation=tf.nn.tanh),
                                         input_keep_prob=keep_prob,
                                         output_keep_prob=keep_prob)
            output_cell = BasicRNNCell(
                num_units=n_outputs,
                activation=tf.nn.tanh
            )
            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(
                [hidden_cell, output_cell]
            )
            self.outputs, _ = tf.nn.dynamic_rnn(
                multi_layer_cell, self.X, dtype=tf.float32
            )

    @classmethod
    def for_training(cls, learning_rate, n_steps, n_inputs, n_hidden,
                     keep_prob=0.5):
        """

        Parameters
        ----------
        learning_rate : float

        n_steps : int
            number of time steps to process at once (unrolled in time)
        n_inputs : int
            number of input values (i.e., number of bins in the STFT)
        n_hidden : int
            number of cells in the hidden layer

        Returns
        -------
        SpeechEncoderModel

        """
        # construct the basic graph...
        model = cls(n_steps, n_inputs, n_hidden, keep_prob)
        # now add training operations...
        model.Y = tf.placeholder(tf.float32,
                                 [None, n_steps, n_inputs], name='labels')
        model.loss = tf.reduce_mean(tf.square(model.outputs - model.Y))
        model.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        model.training_op = model.optimizer.minimize(model.loss)

        model.init = tf.global_variables_initializer()
        return model

    @classmethod
    def for_inference(cls, n_steps, n_inputs, n_hidden):
        """
        Create the model for use when doing inference.
        Parameters
        ----------
        checkpoint_path : str
            filename of the checkpoint file to load when restoring the weights
        n_steps : int
            number of time steps to process at once
        n_inputs : int
            number of input values (i.e., number of bins in the STFT)
        n_hidden : int
            number of cells in the hidden layer

        Returns
        -------
        SpeechEncoderModel

        """
        # construct the basic graph...
        return cls(n_steps, n_inputs, n_hidden)

    def do_inference(self, checkpoint_path, test_file, n_test_cases):
        """
        This is really just a quick test of the inference, useful for
        development, rather than a practical inference service.
        Parameters
        ----------
        checkpoint_path : str
        test_file : str
        n_test_cases : int

        Returns
        -------

        """
        with tf.Session() as sess:
            # load the test data.
            with open(test_file, 'rb') as f:
                t_set = load(f)
                x_test = t_set['X']
                if 'X_org' in t_set.keys():
                    x_org = t_set['X_org']  # if X_org exists, it is the isolated speech.
                else:
                    x_org = x_test  # if no X_org, x_test is actually the isolated speech...no mixed in this set.

            # load the saved variables from the checkpoint file...
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            for ii in range(n_test_cases):
                r = np.random.choice(len(x_test))
                y_hat = self.outputs.eval(
                    feed_dict={
                        self.X: np.expand_dims(np.abs(x_test[r]).T, axis=0)})
                clip = SoundClip.from_stft(x_test[r], 16000, 1024)
                clip.play(blocking=True)
                masked_clip = clip.apply_mask(y_hat[0].T)
                masked_clip.play(blocking=True)
                plt.figure(figsize=(8, 20), tight_layout=True)
                # plot the isolated speech...
                plt.subplot(4, 1, 1)
                plt.pcolormesh(np.log10(np.abs(x_org[r])))
                plt.colorbar()
                plt.title("isolated speech")
                # plot the mixed clip...
                plt.subplot(4, 1, 2)
                qm_x = plt.pcolormesh(np.log10(np.abs(x_test[r])))
                plt.colorbar()
                plt.title("mixed speech + backgound")
                # plot the estimated mask...
                plt.subplot(4, 1, 3)
                plt.pcolormesh(y_hat[0].T, vmin=0.0, vmax=1.0)
                plt.colorbar()
                plt.title("estimated speech mask")
                # plot the masked clip...
                plt.subplot(4, 1, 4)
                plt.pcolormesh(np.log10(np.abs(masked_clip.Zxx)),
                               vmin=qm_x.get_clim()[0],  # set scale the same...
                               vmax=qm_x.get_clim()[1])  # ...as original clip.
                plt.colorbar()
                plt.title("after applying estimated mask")
                plt.tight_layout()
        return

    def do_training(self, training_file, n_epochs, checkpoint_path):
        """
        Setup and perform the training.

        Parameters
        ----------
        training_file : str
            filename of the pickled traing data set (see training_set.py)
        n_epochs : int

        checkpoint_path : str

        Returns
        -------

        """
        # load the training data.
        with open(training_file, 'rb') as f:
            train_set = load(f)
            n_train = int(len(train_set['X']) * 0.8)
            x_train = train_set['X'][:n_train]
            y_train = train_set['Y'][:n_train]
            x_test = train_set['X'][n_train:]
            Y_test = train_set['Y'][n_train:]

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        batch_size = 1
        with tf.Session() as sess:
            self.init.run()
            for epoch in range(n_epochs):
                for ii in range(len(x_train)):
                    x_batch = np.expand_dims(np.abs(x_train[ii]).T, axis=0)
                    y_batch = np.expand_dims(y_train[ii].T, axis=0)
                    sess.run(self.training_op,
                             feed_dict={self.X: x_batch, self.Y: y_batch})
                    print(epoch, ", ", ii, ", ",
                          self.loss.eval(
                              feed_dict={self.X: x_batch, self.Y: y_batch}))

            # Save the variables to disk.
            save_path = saver.save(sess, checkpoint_path)
            print("Model saved in file: %s" % save_path)
            return save_path


def test_training():
    with tf.Graph().as_default():
        model = SpeechEncoderModel.for_training(
            learning_rate=0.0005,
            n_steps=95,  # for a 3 second training clip.
            n_inputs=513,  # for a 1024 point long window
            n_hidden=256,
            keep_prob=0.5
        )
        save_path = model.do_training(
            training_file="/shared/Projects/speech_signal_proc/traing_set_09-21.pkl",
            n_epochs=5,
            checkpoint_path="/shared/Projects/speech_signal_proc/basic_model.ckpt"
        )
        return save_path


def test_inference(checkpoint_path):
    with tf.Graph().as_default():
        model = SpeechEncoderModel.for_inference(
            n_steps=95,  # for a 3 second training clip.
            n_inputs=513,  # for a 1024 point long window
            n_hidden=256
        )
        model.do_inference(
            checkpoint_path=checkpoint_path,
            test_file="/shared/Projects/speech_signal_proc/test_set_09-22.pkl",
            n_test_cases=5
        )
        model.do_inference(
            checkpoint_path=checkpoint_path,
            test_file="/shared/Projects/speech_signal_proc/test_set_mixed_09-25.pkl",
            n_test_cases=3
        )


if __name__ == "__main__":
    checkpoint = test_training()
    #checkpoint = "/shared/Projects/speech_signal_proc/model.ckpt"
    test_inference(checkpoint)
