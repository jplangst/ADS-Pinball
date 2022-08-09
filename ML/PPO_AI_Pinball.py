"""
A simple version of Proximal Policy Optimization (PPO) using single thread and an LSTM layer.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
3. Generalized Advantage Estimation [https://arxiv.org/abs/1506.02438]
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# We are using the compatability mode for TF1 in order to use the RL code. 
# As long as we use this deprecation warnings will occur and fill the output.
# If we upgrade the RL part to use TF2 as well we can remove this.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print("RL Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import numpy as np
import os
from time import time

ENTROPY_BETA = 0.01  # 0.01 for discrete, 0.0 for continuous
LR = 0.0001
MINIBATCH = 32
EPOCHS = 10
EPSILON = 0.1
VF_COEFF = 1.0
L2_REG = 0.001
LSTM_UNITS = 128
LSTM_LAYERS = 1
DENSE_UNITS = 64
KEEP_PROB = 0.8
SIGMA_FLOOR = 0.1  # Useful to set this to a non-zero number since the LSTM can make sigma drop too quickly

class PPO(object):
    def __init__(self, summary_dir="./", gpu=False, greyscale=True, state_dimension=np.array([8]), action_dimension=np.array([5])):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': gpu})
        config.gpu_options.per_process_gpu_memory_fraction = 0.1

        # Modified to fit the Pinball machine
        self.discrete = True 

        self.s_dim = state_dimension #The shape of the state (frames, x.y coordinate) 
        self.a_dim = action_dimension # 0-Noop, 1-LeftTrigger, 2-RightTrigger, 3-BothTriggers, 4-StartButton
        self.actions = tf.placeholder(tf.int32, [None, 1], 'action')

        #This creates CNN layers if result is true, will be false in our case for now
        self.cnn = len(self.s_dim) == 3
        self.greyscale = greyscale  # If not greyscale and using RGB, make sure to divide the images by 255

        self.sess = tf.Session(config=config)
        self.state = tf.placeholder(tf.float32, [None] + list(self.s_dim), 'state')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Use the TensorFlow Dataset API
        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state, "actions": self.actions,
                                                           "rewards": self.rewards, "advantage": self.advantage})
        self.dataset = self.dataset.batch(MINIBATCH, drop_remainder=True)
        self.iterator = self.dataset.make_initializable_iterator()
        batch = self.iterator.get_next()
        self.global_step = tf.train.get_or_create_global_step()

        # Create an old & new policy function but also
        # make separate value & policy functions for evaluation & training (with shared variables)
        pi_old, pi_old_params, _, _ = self._build_anet(batch["state"], 'RLoldpi')
        pi, pi_params, self.pi_i_state, self.pi_f_state = self._build_anet(batch["state"], 'RLpi')
        pi_eval, _, self.pi_eval_i_state, self.evalpi_f_state = self._build_anet(self.state, 'RLpi', reuse=True, batch_size=1)

        vf_old, vf_old_params, _, _ = self._build_cnet(batch["state"], "RLoldvf")
        self.v, vf_params, self.vf_i_state, self.vf_f_state = self._build_cnet(batch["state"], "RLvf")
        self.vf_eval, _, self.vf_eval_i_state, self.vf_eval_f_state = self._build_cnet(self.state, 'RLvf', reuse=True, batch_size=1)

        self.sample_op = tf.squeeze(pi_eval.sample(1), axis=0, name="sample_action")
        self.eval_action = pi_eval.mode()  # Use mode for discrete case. Mode should equal mean in continuous
        self.saver = tf.train.Saver()  # set max_to+keep to keep older checkpoints

        with tf.variable_scope('RLloss'):
            epsilon_decay = tf.train.polynomial_decay(EPSILON, self.global_step, 1e6, 0.01, power=0.0)

            with tf.variable_scope('RLpolicy'):
                # Use floor functions for the probabilities to prevent NaNs when prob = 0
                ratio = tf.maximum(pi.prob(batch["actions"]), 1e-6) / tf.maximum(pi_old.prob(batch["actions"]), 1e-6)
                ratio = tf.clip_by_value(ratio, 0, 10)
                surr1 = batch["advantage"] * ratio
                surr2 = batch["advantage"] * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
                loss_pi = -tf.reduce_mean(tf.minimum(surr1, surr2))
                tf.summary.scalar("loss", loss_pi)

            with tf.variable_scope('RLvalue_function'):
                clipped_value_estimate = vf_old + tf.clip_by_value(self.v - vf_old, -epsilon_decay, epsilon_decay)
                loss_vf1 = tf.squared_difference(clipped_value_estimate, batch["rewards"])
                loss_vf2 = tf.squared_difference(self.v, batch["rewards"])
                loss_vf = tf.reduce_mean(tf.maximum(loss_vf1, loss_vf2)) * 0.5
                # loss_vf = tf.reduce_mean(tf.square(self.v - batch["rewards"])) * 0.5
                tf.summary.scalar("loss", loss_vf)

            with tf.variable_scope('RLentropy'):
                entropy = pi.entropy()
                pol_entpen = -ENTROPY_BETA * tf.reduce_mean(entropy)

            loss = loss_pi + loss_vf * VF_COEFF + pol_entpen
            tf.summary.scalar("total", loss)
            # tf.summary.scalar("epsilon", epsilon_decay)

        with tf.variable_scope('RLtrain'):
            opt = tf.train.AdamOptimizer(LR)
            self.train_op = opt.minimize(loss, global_step=self.global_step, var_list=pi_params + vf_params)

            # Gradient clipping
            # grads, vs = zip(*opt.compute_gradients(loss, var_list=pi_params + vf_params))
            # Need to split the two networks so that clip_by_global_norm works properly
            # pi_grads, pi_vs = grads[:len(pi_params)], vs[:len(pi_params)]
            # vf_grads, vf_vs = grads[len(pi_params):], vs[len(pi_params):]
            # pi_grads, _ = tf.clip_by_global_norm(pi_grads, 10)
            # vf_grads, _ = tf.clip_by_global_norm(vf_grads, 10)
            # self.train_op = opt.apply_gradients(zip(pi_grads + vf_grads, pi_vs + vf_vs), global_step=self.global_step)

            # for grad, var in zip(pi_grads + vf_grads, pi_vs + vf_vs):
            #     tf.summary.histogram(var.name, grad)

        with tf.variable_scope('RLupdate_old'):
            self.update_pi_old_op = [oldp.assign(p) for p, oldp in zip(pi_params, pi_old_params)]
            self.update_vf_old_op = [oldp.assign(p) for p, oldp in zip(vf_params, vf_old_params)]

        self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        tf.summary.scalar("value", tf.reduce_mean(self.v))
        tf.summary.scalar("policy_entropy", tf.reduce_mean(entropy))
        if not self.discrete:
            tf.summary.scalar("sigma", tf.reduce_mean(pi.stddev()))
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    def _build_anet(self, state_in, name, reuse=False, batch_size=MINIBATCH):
        w_reg = None

        with tf.variable_scope(name, reuse=reuse):
            if self.cnn:
                if self.greyscale:
                    state_in = tf.image.rgb_to_grayscale(state_in)
                conv1 = tf.layers.conv2d(inputs=state_in, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
                conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
                conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
                state_in = tf.layers.flatten(conv3)

            l1 = tf.layers.dense(state_in, DENSE_UNITS, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l1")
            l2 = tf.layers.dense(l1, LSTM_UNITS, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l2")

            # LSTM layer
            a_lstm = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_UNITS, name='basic_lstm_cell')
            a_lstm = tf.nn.rnn_cell.DropoutWrapper(a_lstm, output_keep_prob=self.keep_prob)
            a_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[a_lstm] * LSTM_LAYERS)

            a_init_state = a_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
            lstm_in = tf.expand_dims(l2, axis=1)

            a_outputs, a_final_state = tf.nn.dynamic_rnn(cell=a_lstm, inputs=lstm_in, initial_state=a_init_state)
            a_cell_out = tf.reshape(a_outputs, [-1, LSTM_UNITS], name='flatten_lstm_outputs')

            if self.discrete:
                a_logits = tf.layers.dense(a_cell_out, self.a_dim, kernel_regularizer=w_reg, name="pi_logits")
                dist = tf.distributions.Categorical(logits=a_logits)
            else:
                mu = tf.layers.dense(a_cell_out, self.a_dim, tf.nn.tanh, kernel_regularizer=w_reg, name="pi_mu")
                log_sigma = tf.get_variable(name="pi_sigma", shape=self.a_dim, initializer=tf.zeros_initializer())
                dist = tf.distributions.Normal(loc=mu * self.a_bound, scale=tf.maximum(tf.exp(log_sigma), SIGMA_FLOOR))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, params, a_init_state, a_final_state

    def _build_cnet(self, state_in, name, reuse=False, batch_size=MINIBATCH):
        w_reg = tf.keras.regularizers.L2(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
            if self.cnn:
                if self.greyscale:
                    state_in = tf.image.rgb_to_grayscale(state_in)
                conv1 = tf.layers.conv2d(inputs=state_in, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
                conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
                conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
                state_in = tf.layers.flatten(conv3)

            l1 = tf.layers.dense(state_in, DENSE_UNITS, tf.nn.relu, kernel_regularizer=w_reg, name="vf_l1")
            l2 = tf.layers.dense(l1, LSTM_UNITS, tf.nn.relu, kernel_regularizer=w_reg, name="vf_l2")

            # LSTM layer
            c_lstm = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_UNITS, name='basic_lstm_cell')
            c_lstm = tf.nn.rnn_cell.DropoutWrapper(c_lstm, output_keep_prob=self.keep_prob)
            c_lstm = tf.nn.rnn_cell.MultiRNNCell([c_lstm] * LSTM_LAYERS)

            c_init_state = c_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
            lstm_in = tf.expand_dims(l2, axis=1)

            c_outputs, c_final_state = tf.nn.dynamic_rnn(cell=c_lstm, inputs=lstm_in, initial_state=c_init_state)
            c_cell_out = tf.reshape(c_outputs, [-1, LSTM_UNITS], name='flatten_lstm_outputs')

            vf = tf.layers.dense(c_cell_out, 1, kernel_regularizer=w_reg, name="vf_out")
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return vf, params, c_init_state, c_final_state

    def save_model(self, model_path, step=None):
        save_path = self.saver.save(self.sess, os.path.join(model_path, "model.ckpt"), global_step=step)
        return save_path

    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path) #os.path.join(model_path, "model.ckpt"))
        print("Model restored from", model_path)

    def update(self, episode_rollouts):
        start, e_time = time(), []
        self.sess.run([self.update_pi_old_op, self.update_vf_old_op])

        for _ in range(EPOCHS):
            np.random.shuffle(episode_rollouts)
            for ep_s, ep_a, ep_r, ep_adv in episode_rollouts:

                self.sess.run(self.iterator.initializer, feed_dict={self.state: ep_s, self.actions: ep_a,
                                                                    self.rewards: ep_r, self.advantage: ep_adv})

                a_state, c_state = self.sess.run([self.pi_i_state, self.vf_i_state])
                train_ops = [self.summarise, self.global_step, self.pi_f_state, self.vf_f_state, self.train_op]

                while True:
                    try:
                        e_start = time()
                        feed_dict = {self.pi_i_state: a_state, self.vf_i_state: c_state, self.keep_prob: KEEP_PROB}
                        summary, step, a_state, c_state, _ = self.sess.run(train_ops, feed_dict=feed_dict)
                        e_time.append(time() - e_start)
                    except tf.errors.OutOfRangeError:
                        break

        print("Trained in %.3fs. Average %.3fs/minibatch. Global step %i" % (time() - start, np.mean(e_time), step))
        return summary

    def evaluate_state(self, state, lstm_state, stochastic=True):
        if stochastic:
            eval_ops = [self.sample_op, self.vf_eval, self.evalpi_f_state, self.vf_eval_f_state]
        else:
            eval_ops = [self.eval_action, self.vf_eval, self.evalpi_f_state, self.vf_eval_f_state]

        action, value, a_state, c_state = self.sess.run(eval_ops,
                                                        {self.state: state[np.newaxis, :],
                                                         self.pi_eval_i_state: lstm_state[0],
                                                         self.vf_eval_i_state: lstm_state[1],
                                                         self.keep_prob: 1.0})
        return action[0], np.squeeze(value), (a_state, c_state)
