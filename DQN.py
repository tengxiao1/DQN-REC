import numpy as np
import tensorflow as tf
import numpy as np
import os
import argparse
import time
import json
import sys
import re
import datetime


# os.environ['CUDA_VISIBLE_DEVICES']=2



class DQN(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.song_feature_path = "/home/ndir/xiaoteng/reinforcement-learing/data/song_feature.json"
        self.song_style_path = "/home/ndir/xiaoteng/reinforcement-learing/data/song_style.json"
        self.song_era_path = "/home/ndir/xiaoteng/reinforcement-learing/data/song_era.json"
        self.song_lan_path = "/home/ndir/xiaoteng/reinforcement-learing/data/song_lan.json"
        self.song_heat_path = "/home/ndir/xiaoteng/reinforcement-learing/data/song_heat.json"
        self.song_feature_dict, self.song_style_dict,self.song_era_dict, \
          self.song_lan_dict,self.song_heat_dict = self._load_data()

    def _generate_model(self):
        with tf.name_scope("placeholder"):
            with tf.name_scope("user_feature"):
                self.user_tag = tf.placeholder(
                    shape=[None, 1, self.config.user_tag_size],
                    dtype=tf.float32,
                    name="user_tag"
                )
                self.user_lan = tf.placeholder(
                    shape=[None, 1, self.config.user_lan_size],
                    dtype=tf.float32,
                    name="user_lan"
                )
            with tf.name_scope("target_feature"):
                self.target_tag = tf.placeholder(
                    shape=[None, 1, self.config.user_tag_size],
                    dtype=tf.float32,
                    name="target_tag"
                )
                self.target_lan = tf.placeholder(
                    shape=[None, 1, self.config.user_lan_size],
                    dtype=tf.float32,
                    name="target_lan"
                )

            with tf.name_scope("user_continues_feature"):
                self.user_continue_basic_feature = tf.placeholder(tf.float32,
                                                                  shape=(None, 1, self.config.user_continue_basic_dim),
                                                                  name="user_continue_basic_feature")
                self.user_payinfo_feature = tf.placeholder(tf.float32, shape=(None, 1, self.config.user_payinfo_dim),
                                                           name="user_continue_payinfo_feature")
                self.user_buy_feature = tf.placeholder(tf.float32, shape=(None, 1, self.config.user_buy_dim),
                                                       name="user_continue_buy_feature")

            with tf.name_scope("user_state"):
                self.listen_song_feature = tf.placeholder(tf.float32, shape=(None, self.config.hist_length, self.config.song_feature_dim), name="listen_song_feature")
                self.listen_song_style = tf.placeholder(tf.float32, shape=(None, self.config.hist_length, self.config.song_style_dim), name="listen_song_style")
                self.listen_song_era = tf.placeholder(tf.float32,
                                                      shape=(None, self.config.hist_length, self.config.song_era_dim),
                                                      name="listen_song_era")
                self.listen_song_lan = tf.placeholder(tf.float32,
                                                      shape=(None, self.config.hist_length, self.config.song_lan_dim),
                                                      name="listen_song_lan")
                self.listen_song_heat = tf.placeholder(tf.float32,
                                                       shape=(None, self.config.hist_length, self.config.song_heat_dim),
                                                       name="listen_song_heat")
                self.listen_length = tf.placeholder(shape=[None,1], dtype=tf.int32, name="listen_length")

                self.skip_song_feature = tf.placeholder(tf.float32, shape=(
                None, self.config.hist_length, self.config.song_feature_dim), name="skip_song_feature")
                self.skip_song_style = tf.placeholder(tf.float32,
                                                      shape=(None, self.config.hist_length, self.config.song_style_dim),
                                                      name="skip_song_style")
                self.skip_song_era = tf.placeholder(tf.float32,
                                                    shape=(None, self.config.hist_length, self.config.song_era_dim),
                                                    name="skip_song_era")
                self.skip_song_lan = tf.placeholder(tf.float32,
                                                    shape=(None, self.config.hist_length, self.config.song_lan_dim),
                                                    name="skip_song_lan")
                self.skip_song_heat = tf.placeholder(tf.float32,
                                                     shape=(None, self.config.hist_length, self.config.song_heat_dim),
                                                     name="skip_song_heat")
                self.skip_length = tf.placeholder(shape=[None,1], dtype=tf.int32, name="skip_length")

                self.sub_song_feature = tf.placeholder(tf.float32, shape=(
                None, self.config.hist_length, self.config.song_feature_dim), name="sub_song_feature")
                self.sub_song_style = tf.placeholder(tf.float32,
                                                     shape=(None, self.config.hist_length, self.config.song_style_dim),
                                                     name="sub_song_style")
                self.sub_song_era = tf.placeholder(tf.float32,
                                                   shape=(None, self.config.hist_length, self.config.song_era_dim),
                                                   name="sub_song_era")
                self.sub_song_lan = tf.placeholder(tf.float32,
                                                   shape=(None, self.config.hist_length, self.config.song_lan_dim),
                                                   name="sub_song_lan")
                self.sub_song_heat = tf.placeholder(tf.float32,
                                                    shape=(None, self.config.hist_length, self.config.song_heat_dim),
                                                    name="sub_song_heat")
                self.sub_length = tf.placeholder(shape=[None,1], dtype=tf.int32, name="sub_length")

                self.trash_song_feature = tf.placeholder(tf.float32, shape=(
                None, self.config.hist_length, self.config.song_feature_dim), name="trash_song_feature")
                self.trash_song_style = tf.placeholder(tf.float32, shape=(
                None, self.config.hist_length, self.config.song_style_dim), name="trash_song_style")
                self.trash_song_era = tf.placeholder(tf.float32,
                                                     shape=(None, self.config.hist_length, self.config.song_era_dim),
                                                     name="trash_song_era")
                self.trash_song_lan = tf.placeholder(tf.float32,
                                                     shape=(None, self.config.hist_length, self.config.song_lan_dim),
                                                     name="trash_song_lan")
                self.trash_song_heat = tf.placeholder(tf.float32,
                                                      shape=(None, self.config.hist_length, self.config.song_heat_dim),
                                                      name="trash_song_heat")
                self.trash_length = tf.placeholder(shape=[None,1], dtype=tf.int32, name="trash_length")


            with tf.name_scope("song_action"):
                self.song_style = tf.placeholder(shape=[None, 1, self.config.song_style_dim], dtype=tf.float32,
                                                 name="song_style")
                self.song_era = tf.placeholder(shape=[None, 1, self.config.song_era_dim], dtype=tf.float32,
                                               name="song_era")
                self.song_lan = tf.placeholder(shape=[None, 1, self.config.song_lan_dim], dtype=tf.float32,
                                               name="song_lan")
                self.song_heat = tf.placeholder(shape=[None, 1, self.config.song_heat_dim], dtype=tf.float32,
                                                name="song_heat")
                self.song_feature = tf.placeholder(shape=[None, 1, self.config.song_feature_dim], dtype=tf.float32,
                                                   name="song_feature")


            self.max_q = tf.placeholder(shape=[None,1,1], dtype=tf.float32, name="max_q")
            self.reward = tf.placeholder(shape=[None,1,1], dtype=tf.float32, name="reward")

        self.q = self._build_net('eval', True)
        self.q_ = self._build_net('target', False)

        self.global_steps = tf.Variable(0, name="global_steps", dtype=tf.int32, trainable=False)
        self.step_update_op = tf.assign_add(self.global_steps, 1)

        self.param_eval = tf.global_variables('eval')
        self.param_target = tf.global_variables('target')

        self.target_replace_ops = [tf.assign(t, self.config.tau * e + (1 - self.config.tau) * t) for t, e in
                                   zip(self.param_target, self.param_eval)]

        # y = r + gamma * max(q^)
        # y_t
        target_q = self.reward + self.config.gamma * self.max_q

        self.loss = tf.reduce_mean(tf.squared_difference(target_q, self.q))

        self.Optimizer = tf.train.AdamOptimizer(self.config.lr)

        self.train_op = self.Optimizer.minimize(self.loss)



    def _attention_positive(self, user_hist_input, song_input, length,trainable):
        """
        :param user_hist_input: [B, H, S]
        :param song_input: [B, S]
        :return: [B, 1, S]
        """
        embedding_dim = song_input.get_shape().as_list()[-1]
        hist_length = user_hist_input.get_shape().as_list()[1]
        queries = tf.reshape(tf.tile(song_input, [1, 1, hist_length]), (-1, hist_length, embedding_dim), name="repeat_song_input")

        attention_input = tf.concat([queries, user_hist_input, queries * user_hist_input], axis=-1, name="attention_input")
        att_1 = tf.layers.dense(attention_input, units=80, activation=tf.nn.relu, name="att_1", trainable=trainable,reuse=tf.AUTO_REUSE)
        att_2 = tf.layers.dense(att_1, units=1, activation=None, name="att_3", trainable=trainable,reuse=tf.AUTO_REUSE)

        # add mask
        attention_out = tf.reshape(att_2, (-1, 1, hist_length), name="attention_out")
        key_mask = tf.sequence_mask(length, hist_length)
        padding = tf.ones_like(attention_out, dtype=tf.float32)* (-2 ** 16 + 1)
        att_out = tf.where(key_mask, attention_out, padding)

        att_out = att_out / (user_hist_input.get_shape().as_list()[-1]**0.5)
        att_out = tf.nn.softmax(att_out)

        weight_sum_pooling = tf.matmul(att_out, user_hist_input, name="weighted_sum")
        return weight_sum_pooling

    def _attention_negative(self, user_hist_input, song_input,length,trainable):
        """
        :param user_hist_input: [B, H, S]
        :param song_input: [B, S]
        :return: [B, 1, S]
        """

        embedding_dim = song_input.get_shape().as_list()[-1]
        hist_length = user_hist_input.get_shape().as_list()[1]
        queries = tf.reshape(tf.tile(song_input, [1, 1, hist_length]), (-1, hist_length, embedding_dim), name="repeat_song_input")

        attention_input = tf.concat([queries, user_hist_input, queries * user_hist_input], axis=-1, name="attention_input")
        att_1 = tf.layers.dense(attention_input, units=80, activation=tf.nn.relu, name="att_1",  trainable=trainable,reuse=tf.AUTO_REUSE)
        att_2 = tf.layers.dense(att_1, units=1, activation=None, name="att_3", trainable=trainable,reuse=tf.AUTO_REUSE)

        # add mask
        attention_out = tf.reshape(att_2, (-1, 1, hist_length), name="attention_out")
        key_mask = tf.sequence_mask(length, hist_length)
        padding = tf.ones_like(attention_out, dtype=tf.float32)* (-2 ** 16 + 1)
        att_out = tf.where(key_mask, attention_out, padding)

        att_out = att_out / (user_hist_input.get_shape().as_list()[-1]**0.5)
        att_out = tf.nn.softmax(att_out)

        weight_sum_pooling = tf.matmul(att_out, user_hist_input, name="weighted_sum")
        return weight_sum_pooling

    def _load_data(self):
        print("load start")
        song_feature_dict = json.load(open(self.song_feature_path, 'r'))
        print("load song feature finished")
        song_style_dict = json.load(open(self.song_style_path, 'r'))
        print("load song style finished")
        song_era_dict = json.load(open(self.song_era_path, 'r'))
        print("load song era finished")
        song_lan_dict = json.load(open(self.song_lan_path, 'r'))
        print("load song lan finished")
        song_heat_dict = json.load(open(self.song_heat_path, 'r'))
        #song_feature_dict={}
        #song_style_dict={}
        #song_era_dict={}
        #song_lan_dict={}
        #song_heat_dict={}
        print("load finished")
        return song_feature_dict, song_style_dict, song_era_dict, song_lan_dict, song_heat_dict

    def _build_net(self, scope, trainable):  # Q-net
        with tf.variable_scope(scope):  # V(s)
            with tf.name_scope("variable"):
                song_style_embedding = tf.get_variable(
                    shape=[self.config.song_style_dim, self.config.embedding_size],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer,
                    trainable=trainable,
                    name="song_style_embedding"
                )

                song_era_embedding = tf.get_variable(
                    shape=[self.config.song_era_dim, self.config.embedding_size],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer,
                    trainable=trainable,
                    name="song_era_embedding"
                )

                song_lan_embedding = tf.get_variable(
                    shape=[self.config.song_lan_dim, self.config.embedding_size],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer,
                    trainable=trainable,
                    name="song_lan_embedding"
                )

                song_heat_embedding = tf.get_variable(
                    shape=[self.config.song_heat_dim, self.config.embedding_size],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer,
                    trainable=trainable,
                    name="song_heat_embedding"
                )

                user_tag_embedding = tf.get_variable(shape=[self.config.user_tag_size, self.config.embedding_size],
                                                     name="user_tag_embedding",
                                                     initializer=tf.random_uniform_initializer,
                                                     trainable=trainable,
                                                     dtype=tf.float32)
                user_lan_embedding = tf.get_variable(shape=[self.config.user_lan_size, self.config.embedding_size],
                                                     name="user_lan_embedding",
                                                     initializer=tf.random_uniform_initializer,
                                                     trainable=trainable,
                                                     dtype=tf.float32)
            with tf.name_scope("input_Q-Net"):
                with tf.name_scope("user_state_input"):
                    listen_song_style_input = tf.tensordot(
                        tf.reshape(self.listen_song_style, shape=(-1, 10, self.config.song_style_dim)),
                        song_style_embedding, axes=1, name="listen_song_style_input")
                    listen_song_era_input = tf.tensordot(
                        tf.reshape(self.listen_song_era, shape=(-1, 10, self.config.song_era_dim)), song_era_embedding,
                        axes=1, name="listen_song_era_input")
                    listen_song_lan_input = tf.tensordot(
                        tf.reshape(self.listen_song_lan, shape=(-1, 10, self.config.song_lan_dim)), song_lan_embedding,
                        axes=1, name="listen_song_lan_input")
                    listen_song_heat_input = tf.tensordot(
                        tf.reshape(self.listen_song_heat, shape=(-1, 10, self.config.song_heat_dim)),
                        song_heat_embedding, axes=1, name="listen_song_heat_input")

                    skip_song_style_input = tf.tensordot(
                        tf.reshape(self.skip_song_style, shape=(-1, 10, self.config.song_style_dim)),
                        song_style_embedding, axes=1, name="skip_song_style_input")
                    skip_song_era_input = tf.tensordot(
                        tf.reshape(self.skip_song_era, shape=(-1, 10, self.config.song_era_dim)), song_era_embedding,
                        axes=1, name="skip_song_era_input")
                    skip_song_lan_input = tf.tensordot(
                        tf.reshape(self.skip_song_lan, shape=(-1, 10, self.config.song_lan_dim)), song_lan_embedding,
                        axes=1, name="skip_song_lan_input")
                    skip_song_heat_input = tf.tensordot(
                        tf.reshape(self.skip_song_heat, shape=(-1, 10, self.config.song_heat_dim)), song_heat_embedding,
                        axes=1, name="skip_song_heat_input")

                    sub_song_style_input = tf.tensordot(
                        tf.reshape(self.sub_song_style, shape=(-1, 10, self.config.song_style_dim)),
                        song_style_embedding, axes=1, name="sub_song_style_input")
                    sub_song_era_input = tf.tensordot(
                        tf.reshape(self.sub_song_era, shape=(-1, 10, self.config.song_era_dim)), song_era_embedding,
                        axes=1, name="sub_song_era_input")
                    sub_song_lan_input = tf.tensordot(
                        tf.reshape(self.sub_song_lan, shape=(-1, 10, self.config.song_lan_dim)), song_lan_embedding,
                        axes=1, name="sub_song_lan_input")
                    sub_song_heat_input = tf.tensordot(
                        tf.reshape(self.sub_song_heat, shape=(-1, 10, self.config.song_heat_dim)), song_heat_embedding,
                        axes=1, name="sub_song_heat_input")

                    trash_song_style_input = tf.tensordot(
                        tf.reshape(self.trash_song_style, shape=(-1, 10, self.config.song_style_dim)),
                        song_style_embedding, axes=1, name="trash_song_style_input")
                    trash_song_era_input = tf.tensordot(
                        tf.reshape(self.trash_song_era, shape=(-1, 10, self.config.song_era_dim)), song_era_embedding,
                        axes=1, name="trash_song_era_input")
                    trash_song_lan_input = tf.tensordot(
                        tf.reshape(self.trash_song_lan, shape=(-1, 10, self.config.song_lan_dim)), song_lan_embedding,
                        axes=1, name="trash_song_lan_input")
                    trash_song_heat_input = tf.tensordot(
                        tf.reshape(self.trash_song_heat, shape=(-1, 10, self.config.song_heat_dim)),
                        song_heat_embedding, axes=1, name="trash_song_heat_input") #[B,10,dim]

                    hist_listen_input = tf.concat([self.listen_song_feature, listen_song_style_input,
                                                  listen_song_era_input, listen_song_lan_input, listen_song_heat_input],
                                                  axis=-1,
                                                  name="hist_listen_input")
                    hist_skip_input = tf.concat([self.skip_song_feature, skip_song_style_input, skip_song_era_input,
                                                skip_song_lan_input, skip_song_heat_input], axis=-1,
                                                name="hist_skip_input")
                    hist_sub_input = tf.concat([self.sub_song_feature, sub_song_style_input, sub_song_era_input,
                                               sub_song_lan_input, sub_song_heat_input], axis=-1, name="hist_sub_input")
                    hist_trash_input = tf.concat([self.trash_song_feature, trash_song_style_input, trash_song_era_input,
                                                 trash_song_lan_input, trash_song_heat_input], axis=-1,
                                                 name="hist_trash_input")

                song_style_input = tf.tensordot(self.song_style, song_style_embedding, axes=1, name="song_style_input")
                song_era_input = tf.tensordot(self.song_era, song_era_embedding, axes=1, name="song_era_input")
                song_lan_input = tf.tensordot(self.song_lan, song_lan_embedding, axes=1, name="song_lan_input")
                song_heat_input = tf.tensordot(self.song_heat, song_heat_embedding, axes=1, name="song_heat_input")
                song_input = tf.concat(
                    [song_style_input, song_era_input, song_lan_input, song_heat_input, self.song_feature], axis=-1,
                    name="song_input")
                attention_skip = self._attention_negative(hist_skip_input, song_input,self.skip_length,trainable)
                attention_listen = self._attention_positive(hist_listen_input, song_input,self.listen_length,trainable)
                attention_trash = self._attention_negative(hist_trash_input, song_input,self.trash_length,trainable)
                attention_sub = self._attention_positive(hist_sub_input, song_input,self.sub_length,trainable)

                cross__lan_feature=tf.multiply(self.user_lan,self.target_lan)
                cross__tag_feature=tf.multiply(self.user_tag,self.target_tag)


                hist_input = tf.concat(
                    [attention_skip, attention_listen, attention_trash, attention_sub], axis=-1)

                user_tag_input = tf.tensordot(self.user_tag, user_tag_embedding, axes=1, name="user_tag_input")
                user_lan_input = tf.tensordot(self.user_lan, user_lan_embedding, axes=1, name="user_lan_input")
                user_input = tf.concat(
                    [self.user_continue_basic_feature, self.user_payinfo_feature, self.user_buy_feature, user_tag_input,
                     user_lan_input], axis=-1, name="user_input")

                state_input = tf.concat(
                    [user_input, hist_input, song_input, cross__lan_feature, cross__tag_feature], axis=-1)
            with tf.variable_scope('Q-net'):
                Q_layer0 = tf.layers.dense(state_input, 100, activation=tf.nn.relu, trainable=trainable,
                                           kernel_initializer=tf.random_normal_initializer(0., 0.1),
                                           bias_initializer=tf.constant_initializer(.1))
                Q_layer1 = tf.layers.dense(Q_layer0, 50, activation=tf.nn.relu, trainable=trainable,
                                           kernel_initializer=tf.random_normal_initializer(0., 0.1),
                                           bias_initializer=tf.constant_initializer(.1))
                Q_layer2 = tf.layers.dense(Q_layer1, 20, activation=tf.nn.relu, trainable=trainable,
                                           kernel_initializer=tf.random_normal_initializer(0., 0.1),
                                           bias_initializer=tf.constant_initializer(.1))
                q_output = tf.layers.dense(Q_layer2, 1, name='q', trainable=trainable,
                                           kernel_initializer=tf.random_normal_initializer(0., 0.1),
                                           bias_initializer=tf.constant_initializer(.1))
        return q_output

    def get_max_q(self, next_train_batch, song_features, song_styles, song_eras, song_lans, song_heats,
                  next_skip_song_feature, next_skip_song_style, next_skip_song_era, next_skip_song_lan,
                  next_skip_song_heat,
                  next_listen_song_feature, next_listen_song_style, next_listen_song_era, next_listen_song_lan,
                  next_listen_song_heat,
                  next_sub_song_feature, next_sub_song_style, next_sub_song_era, next_sub_song_lan, next_sub_song_heat,
                  next_trash_song_feature, next_trash_song_style, next_trash_song_era, next_trash_song_lan,
                  next_trash_song_heat,candidates_num,i):



        q_ = self.sess.run([self.q], feed_dict={
                self.user_continue_basic_feature: np.tile(next_train_batch["features"][:, :, 0:53][i,:][np.newaxis,:],(candidates_num,1,1)),  # 53
                self.user_tag: np.tile(next_train_batch["features"][:, :, 364:381][i,:][np.newaxis,:],(candidates_num,1,1)),  # 17
                self.user_lan: np.tile(next_train_batch["features"][:, :, 381:387][i,:][np.newaxis,:],(candidates_num,1,1)),  # 6
                self.target_tag: np.tile(next_train_batch["features"][:, :, 410:427][i,:][np.newaxis,:],(candidates_num,1,1)),  # 17
                self.target_lan: np.tile(next_train_batch["features"][:, :, 427:433][i,:][np.newaxis,:],(candidates_num,1,1)),  # 6
                self.user_payinfo_feature: np.tile(next_train_batch["features"][:, :, 459:467][i,:][np.newaxis,:],(candidates_num,1,1)),  # 8
                self.user_buy_feature: np.tile(next_train_batch["features"][:, :, 475:483][i,:][np.newaxis,:],(candidates_num,1,1)),  # 8 [candidates_num,1,dim]

                self.song_style: song_styles,
                self.song_era: song_eras,
                self.song_lan: song_lans,
                self.song_heat: song_heats,
                self.song_feature: song_features, #[candidates_num,1,dim]

                self.skip_song_feature: np.tile(next_skip_song_feature,(candidates_num,1,1)),
                self.skip_song_style: np.tile(next_skip_song_style,(candidates_num,1,1)),
                self.skip_song_era: np.tile(next_skip_song_era,(candidates_num,1,1)),
                self.skip_song_lan: np.tile(next_skip_song_lan,(candidates_num,1,1)),
                self.skip_song_heat: np.tile(next_skip_song_heat,(candidates_num,1,1)),
                self.skip_length: np.tile(next_train_batch["next_skip_len"][i,:][np.newaxis,:],(candidates_num,1)),

                self.listen_song_feature: np.tile(next_listen_song_feature,(candidates_num,1,1)),
                self.listen_song_style: np.tile(next_listen_song_style,(candidates_num,1,1)),
                self.listen_song_era: np.tile(next_listen_song_era,(candidates_num,1,1)),
                self.listen_song_lan: np.tile(next_listen_song_lan,(candidates_num,1,1)),
                self.listen_song_heat: np.tile(next_listen_song_heat,(candidates_num,1,1)), ##[candidates_num,hist_len,dim]
                self.listen_length: np.tile(next_train_batch["next_play_end_len"][i,:][np.newaxis,:],(candidates_num,1)),##[candidates_num,1]

                self.sub_song_feature: np.tile(next_sub_song_feature,(candidates_num,1,1)),
                self.sub_song_style: np.tile(next_sub_song_style,(candidates_num,1,1)),
                self.sub_song_era: np.tile(next_sub_song_era,(candidates_num,1,1)),
                self.sub_song_lan: np.tile(next_sub_song_lan,(candidates_num,1,1)),
                self.sub_song_heat: np.tile(next_sub_song_heat,(candidates_num,1,1)),
                self.sub_length: np.tile(next_train_batch["next_red_seq_len"][i,:][np.newaxis,:],(candidates_num,1)),

                self.trash_song_feature: np.tile(next_trash_song_feature,(candidates_num,1,1)),
                self.trash_song_style: np.tile(next_trash_song_style,(candidates_num,1,1)),
                self.trash_song_era: np.tile(next_trash_song_era,(candidates_num,1,1)),
                self.trash_song_lan: np.tile(next_trash_song_lan,(candidates_num,1,1)),
                self.trash_song_heat: np.tile(next_trash_song_heat,(candidates_num,1,1)),
                self.trash_length: np.tile(next_train_batch["next_trash_seq_len"][i,:][np.newaxis,:],(candidates_num,1)),

            })

        return  q_

    def get_grident(self, next_train_batch, song_feature, song_style, song_era, song_lan, song_heat,
                    skip_song_feature, skip_song_style, skip_song_era, skip_song_lan, skip_song_heat,
                    listen_song_feature, listen_song_style, listen_song_era, listen_song_lan, listen_song_heat,
                    sub_song_feature, sub_song_style, sub_song_era, sub_song_lan, sub_song_heat,
                    trash_song_feature, trash_song_style, trash_song_era, trash_song_lan, trash_song_heat, max_q):
        is_playend = np.reshape(next_train_batch["is_playend"], -1)
        is_direct_buy = np.reshape(next_train_batch["is_direct_buy"], -1)
        is_click_sku = np.reshape(next_train_batch["is_click_sku"], -1)
        is_success_buy = np.reshape(next_train_batch["is_success_buy"], -1)
        reward = np.zeros(len(is_playend),dtype=float)
        for i in range(len(is_playend)):
            if is_direct_buy[i] == 1.0 or is_success_buy[i] == 1.0:
                reward[i] = 10.0
                continue
            elif is_click_sku[i] == 1.0:
                reward[i] = 5.0
                continue
            elif is_playend[i] == 1.0:
                reward[i] = 1.0
                continue
            else:
                reward[i] = -1.0

        cur_step,  train_op, loss = self.sess.run([self.step_update_op, self.train_op, self.loss], feed_dict={
            self.user_continue_basic_feature: next_train_batch["features"][:, :, 0:53],  # 53
            self.user_tag: next_train_batch["features"][:, :, 364:381],  # 17
            self.user_lan: next_train_batch["features"][:, :, 381:387],  # 6
            self.target_tag: next_train_batch["features"][:, :, 410:427],  # 17
            self.target_lan: next_train_batch["features"][:, :, 427:433],  # 6
            self.user_payinfo_feature: next_train_batch["features"][:, :, 459:467],  # 8
            self.user_buy_feature: next_train_batch["features"][:, :, 475:483],  # 8

            self.song_style: song_style,
            self.song_era: song_era,
            self.song_lan: song_lan,
            self.song_heat: song_heat,
            self.song_feature: song_feature,

            self.skip_song_feature: skip_song_feature,
            self.skip_song_style: skip_song_style,
            self.skip_song_era: skip_song_era,
            self.skip_song_lan: skip_song_lan,
            self.skip_song_heat: skip_song_heat,
            self.skip_length: next_train_batch["pre_skip_len"],

            self.listen_song_feature: listen_song_feature,
            self.listen_song_style: listen_song_style,
            self.listen_song_era: listen_song_era,
            self.listen_song_lan: listen_song_lan,
            self.listen_song_heat: listen_song_heat,
            self.listen_length: next_train_batch["pre_play_end_len"],

            self.sub_song_feature: sub_song_feature,
            self.sub_song_style: sub_song_style,
            self.sub_song_era: sub_song_era,
            self.sub_song_lan: sub_song_lan,
            self.sub_song_heat: sub_song_heat,
            self.sub_length: next_train_batch["pre_red_seq_len"],

            self.trash_song_feature: trash_song_feature,
            self.trash_song_style: trash_song_style,
            self.trash_song_era: trash_song_era,
            self.trash_song_lan: trash_song_lan,
            self.trash_song_heat: trash_song_heat,
            self.trash_length: next_train_batch["pre_trash_seq_len"],

            self.max_q: max_q.reshape(len(is_playend),1,1),
            self.reward: reward.reshape(len(is_playend),1,1),

        })
        return cur_step,train_op, loss

    def backward(self, grads):
        grads_sum = {}
        for i in range(len(self.grads_holder)):
            k = self.grads_holder[i][0]

            grads_sum[k] = sum([g[i][0] for g in grads])

        self.sess.run(self.optm, feed_dict=grads_sum)

    def update(self):
        self.sess.run(self.target_replace_ops)

    def count_weight_num(self):
        print ("network parameters-----------")
        for v in tf.all_variables():
            print (v.get_shape)
        print ("the number of net parameters--")
        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.all_variables()]))

    def _load_file(self, file_dir):
        train_files = [file.split()[-1] for file in os.popen("hadoop fs -ls {}".format(file_dir))]

        train_files = list(filter(lambda file: re.match(".*part.*", file), train_files))
        return train_files

    def _feature_parse(self, serialized_example):
        data = {
            "user_id": tf.FixedLenFeature([1], tf.string),
            "song_id": tf.FixedLenFeature([1], tf.string),
            "date": tf.FixedLenFeature([1], tf.string),
            "is_playend": tf.FixedLenFeature([1], tf.string),
            "is_buy": tf.FixedLenFeature([1], tf.string),
            "is_direct_buy": tf.FixedLenFeature([1], tf.float32),
            "is_click_sku": tf.FixedLenFeature([1], tf.float32),
            "is_click_buy": tf.FixedLenFeature([1], tf.float32),
            "is_success_buy": tf.FixedLenFeature([1], tf.float32),
            "features": tf.FixedLenFeature([1, self.config.features_size], tf.float32),
            "pre_play_end_seq": tf.FixedLenFeature([self.config.hist_length], tf.string),
            "pre_play_end_len": tf.FixedLenFeature([1], tf.int64),
            "pre_skip_seq": tf.FixedLenFeature([self.config.hist_length], tf.string),
            "pre_skip_len": tf.FixedLenFeature([1], tf.int64),
            "pre_red_seq": tf.FixedLenFeature([self.config.hist_length], tf.string),
            "pre_red_seq_len": tf.FixedLenFeature([1], tf.int64),
            "pre_trash_seq": tf.FixedLenFeature([self.config.hist_length], tf.string),
            "pre_trash_seq_len": tf.FixedLenFeature([1], tf.int64),

            "next_skip_seq": tf.FixedLenFeature([self.config.hist_length], tf.string),
            "next_skip_len": tf.FixedLenFeature([1], tf.int64),
            "next_play_end_seq": tf.FixedLenFeature([self.config.hist_length], tf.string),
            "next_play_end_len": tf.FixedLenFeature([1], tf.int64),
            "next_red_seq": tf.FixedLenFeature([self.config.hist_length], tf.string),
            "next_red_seq_len": tf.FixedLenFeature([1], tf.int64),
            "next_trash_seq": tf.FixedLenFeature([self.config.hist_length], tf.string),
            "next_trash_seq_len": tf.FixedLenFeature([1], tf.int64),

            "candidate_songId_new": tf.FixedLenFeature([1], tf.string),

            "candidate_songId": tf.FixedLenFeature([1], tf.string)

        }
        batch = tf.parse_example(serialized_example, features=data)
        return batch

    def _train_iter(self, train_files, parallels):
        dataset = tf.data.TFRecordDataset(train_files, num_parallel_reads=parallels)
        dataset = dataset.batch(self.config.train_batch_size) \
            .shuffle(500) \
            .map(lambda x: self._feature_parse(x)) \
            .prefetch(2)
        iteration = dataset.make_initializable_iterator()
        return iteration




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="allocate gpu id", default="7")
    parser.add_argument("--lr", help="learning rate", default=0.001)
    parser.add_argument("--epochs", help="training epochs", default=1)
    parser.add_argument("--train_batch_size", help="train_batch_size", default=256)

    parser.add_argument("--features_size", help="features_size", default=483)
    parser.add_argument("--user_continue_basic_dim", help="user_continue_basic_dim", default=53)
    parser.add_argument("--user_payinfo_dim", help="user_payinfo_dim", default=8)
    parser.add_argument("--user_buy_dim", help="user_buy_dim", default=8)
    parser.add_argument("--user_tag_size", help="user_tag_size", default=17)
    parser.add_argument("--user_lan_size", help="user_lan_size", default=6)

    parser.add_argument("--song_style_dim", help="song_style_dim", default=19)
    parser.add_argument("--song_era_dim", help="song_era_dim", default=8)#8
    parser.add_argument("--song_lan_dim", help="song_lan_dim", default=6)#6
    parser.add_argument("--song_heat_dim", help="song_heat_dim", default=9)#9
    parser.add_argument("--song_feature_dim", help="song_feature_dim", default=100)
    parser.add_argument("--embedding_size", help="embedding size", default=50)

    parser.add_argument("--hist_length", help="hist_length", default=10)
    parser.add_argument("--candidate_length",help="candidate_length",default=300)
    parser.add_argument("--avg_feature_len", help="avg feature len", default=200)

    parser.add_argument("--parallels", help="parallels reader", default=8)
    parser.add_argument("--gamma", help="gamma", default=0.9)
    parser.add_argument("--tau", help="update target ratio", default=0.01)

    parser.add_argument("--train_dir", help="train file dir", required=True)
    # parser.add_argument("--test_dir", help="test file dir", required=True)

    parser.add_argument("--model_path", help="model path", required=True)
    parser.add_argument("--version", help="version", default="v1")
    parser.add_argument("--training_flag", help="training_flag", default=True)

    config = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        dqn = DQN(sess, config)
        dqn._generate_model()
        sess.run(tf.global_variables_initializer())
        print("start----------")
        print("start time", time.strftime("%H:%M:%S", time.localtime()))
        saver = tf.train.Saver(max_to_keep=10)
        train_files = dqn._load_file(dqn.config.train_dir)
        for epoch in range(dqn.config.epochs):
            train_batch = dqn._train_iter(train_files, dqn.config.parallels)
            sess.run(train_batch.initializer)
            next_element = train_batch.get_next()
            default_song_feature = [0.0] * dqn.config.song_feature_dim
            default_song_style = [0.0] * dqn.config.song_style_dim
            default_song_era = [0.0] * dqn.config.song_era_dim
            default_song_lan = [0.0] * dqn.config.song_lan_dim
            default_song_heat = [0.0] * dqn.config.song_heat_dim
            iteration = 1
            batch_loss = 0
            dqn.count_weight_num()

            while True:
                try:
                    max_q_batch=[]
                    starttime = datetime.datetime.now()
                    next_train_batch = sess.run(next_element)
                    for i in range(next_train_batch["next_skip_seq"].shape[0]):
                        next_skip_song_feature= np.array(list(
                            map(lambda seq: [dqn.song_feature_dict.setdefault(s, default_song_feature) for s in seq],
                                next_train_batch["next_skip_seq"][i,:][np.newaxis,:])))
                        next_skip_song_style = np.array(list(
                            map(lambda seq: [dqn.song_style_dict.setdefault(s, default_song_style) for s in seq],
                                next_train_batch["next_skip_seq"][i,:][np.newaxis,:])))
                        next_skip_song_era = np.array(list(
                            map(lambda seq: [dqn.song_era_dict.setdefault(s, default_song_era) for s in seq],
                                next_train_batch["next_skip_seq"][i,:][np.newaxis,:])))
                        next_skip_song_lan = np.array(list(
                            map(lambda seq: [dqn.song_lan_dict.setdefault(s, default_song_lan) for s in seq],
                                next_train_batch["next_skip_seq"][i,:][np.newaxis,:])))
                        next_skip_song_heat = np.array(list(
                            map(lambda seq: [dqn.song_heat_dict.setdefault(s, default_song_heat) for s in seq],
                                next_train_batch["next_skip_seq"][i,:][np.newaxis,:])))
                        next_listen_song_feature = np.array(list(
                            map(lambda seq: [dqn.song_feature_dict.setdefault(s, default_song_feature) for s in seq],
                                next_train_batch["next_play_end_seq"][i,:][np.newaxis,:])))
                        next_listen_song_style = np.array(list(
                            map(lambda seq: [dqn.song_style_dict.setdefault(s, default_song_style) for s in seq],
                                next_train_batch["next_play_end_seq"][i,:][np.newaxis,:])))
                        next_listen_song_era = np.array(list(
                            map(lambda seq: [dqn.song_era_dict.setdefault(s, default_song_era) for s in seq],
                                next_train_batch["next_play_end_seq"][i,:][np.newaxis,:])))
                        next_listen_song_lan = np.array(list(
                            map(lambda seq: [dqn.song_lan_dict.setdefault(s, default_song_lan) for s in seq],
                                next_train_batch["next_play_end_seq"][i,:][np.newaxis,:])))
                        next_listen_song_heat = np.array(list(
                            map(lambda seq: [dqn.song_heat_dict.setdefault(s, default_song_heat) for s in seq],
                                next_train_batch["next_play_end_seq"][i,:][np.newaxis,:])))

                        next_sub_song_feature = np.array(list(
                            map(lambda seq: [dqn.song_feature_dict.setdefault(s, default_song_feature) for s in seq],
                                next_train_batch["next_red_seq"][i,:][np.newaxis,:])))
                        next_sub_song_style = np.array(list(
                            map(lambda seq: [dqn.song_style_dict.setdefault(s, default_song_style) for s in seq],
                                next_train_batch["next_red_seq"][i,:][np.newaxis,:])))
                        next_sub_song_era = np.array(list(
                            map(lambda seq: [dqn.song_era_dict.setdefault(s, default_song_era) for s in seq],
                                next_train_batch["next_red_seq"][i,:][np.newaxis,:])))
                        next_sub_song_lan = np.array(list(
                            map(lambda seq: [dqn.song_lan_dict.setdefault(s, default_song_lan) for s in seq],
                                next_train_batch["next_red_seq"][i,:][np.newaxis,:])))
                        next_sub_song_heat = np.array(list(
                            map(lambda seq: [dqn.song_heat_dict.setdefault(s, default_song_heat) for s in seq],
                                next_train_batch["next_red_seq"][i,:][np.newaxis,:])))

                        next_trash_song_feature = np.array(list(
                            map(lambda seq: [dqn.song_feature_dict.setdefault(s, default_song_feature) for s in seq],
                                next_train_batch["next_trash_seq"][i,:][np.newaxis,:])))
                        next_trash_song_style = np.array(list(
                            map(lambda seq: [dqn.song_style_dict.setdefault(s, default_song_style) for s in seq],
                                next_train_batch["next_trash_seq"][i,:][np.newaxis,:])))
                        next_trash_song_era = np.array(list(
                            map(lambda seq: [dqn.song_era_dict.setdefault(s, default_song_era) for s in seq],
                                next_train_batch["next_trash_seq"][i,:][np.newaxis,:])))
                        next_trash_song_lan = np.array(list(
                            map(lambda seq: [dqn.song_lan_dict.setdefault(s, default_song_lan) for s in seq],
                                next_train_batch["next_trash_seq"][i,:][np.newaxis,:])))
                        next_trash_song_heat = np.array(list(
                            map(lambda seq: [dqn.song_heat_dict.setdefault(s, default_song_heat) for s in seq],
                                next_train_batch["next_trash_seq"][i,:][np.newaxis,:])))

                        candidate_songId_new=np.reshape(next_train_batch["candidate_songId_new"][i,:][np.newaxis,:], -1).tolist()[0]

                        candidate_songId=np.reshape(next_train_batch["candidate_songId"][i,:][np.newaxis,:], -1).tolist()[0]
                        if candidate_songId_new == "-1":
                            max_q_batch.append(0.0)
                            continue

                        candidate_songId = np.array(candidate_songId.split(","))

                        candidates = np.append(candidate_songId_new, candidate_songId)

                        candidates_num=0
                        song_features=[]
                        song_styles=[]
                        song_eras=[]
                        song_lans=[]
                        song_heats=[]
                        starttime1 = datetime.datetime.now()
                        for id in candidates:
                            if id=="-1":
                                continue
                            song_feature = np.array(list(dqn.song_feature_dict.setdefault(id, default_song_feature)))[
                                  np.newaxis, :]
                            song_style = np.array(list(dqn.song_style_dict.setdefault(id,default_song_style)))[np.newaxis, :] #[1,n]
                            song_era = np.array(list(dqn.song_era_dict.setdefault(id,default_song_era)))[np.newaxis, :]
                            song_lan = np.array(list(dqn.song_lan_dict.setdefault(id,default_song_lan)))[np.newaxis, :]
                            song_heat = np.array(list(dqn.song_heat_dict.setdefault(id,default_song_heat)))[np.newaxis, :]
                            if candidates_num==0:
                                song_features=song_feature
                                song_styles=song_style
                                song_eras=song_era
                                song_lans=song_lan
                                song_heats=song_heat
                            else:
                                song_features=np.concatenate((song_features,song_feature),axis=0)
                                song_styles = np.concatenate((song_styles, song_style), axis=0)
                                song_eras=np.concatenate((song_eras,song_era),axis=0)
                                song_lans=np.concatenate((song_lans,song_lan),axis=0)
                                song_heats=np.concatenate((song_heats,song_heat),axis=0)
                            candidates_num+=1


                        song_features=song_features[:,np.newaxis,:]
                        song_styles=song_styles[:,np.newaxis,:]
                        song_eras=song_eras[:,np.newaxis,:]
                        song_lans=song_lans[:,np.newaxis,:]
                        song_heats=song_heats[:,np.newaxis,:]  #[candidates_num, 1, dim]
                        candidate_q = dqn.get_max_q(next_train_batch, song_features, song_styles, song_eras, song_lans,
                                                          song_heats,next_skip_song_feature, next_skip_song_style, next_skip_song_era,
                                                          next_skip_song_lan, next_skip_song_heat,
                                                          next_listen_song_feature, next_listen_song_style,
                                                          next_listen_song_era, next_listen_song_lan, next_listen_song_heat,
                                                          next_sub_song_feature, next_sub_song_style, next_sub_song_era,
                                                          next_sub_song_lan, next_sub_song_heat,
                                                          next_trash_song_feature, next_trash_song_style,
                                                          next_trash_song_era, next_trash_song_lan, next_trash_song_heat,candidates_num,i
                                                          )
                        candidate_q=np.reshape(candidate_q,-1)
                        max_q=candidate_q.max()
                        max_q_batch.append(max_q)


                    skip_song_feature = np.array(list(
                        map(lambda seq: [dqn.song_feature_dict.setdefault(s, default_song_feature) for s in seq],
                            next_train_batch["pre_skip_seq"])))

                    skip_song_style = np.array(list(
                        map(lambda seq: [dqn.song_style_dict.setdefault(s, default_song_style) for s in seq],
                            next_train_batch["pre_skip_seq"])))
                    skip_song_era = np.array(list(
                        map(lambda seq: [dqn.song_era_dict.setdefault(s, default_song_era) for s in seq],
                            next_train_batch["pre_skip_seq"])))
                    skip_song_lan = np.array(list(
                        map(lambda seq: [dqn.song_lan_dict.setdefault(s, default_song_lan) for s in seq],
                            next_train_batch["pre_skip_seq"])))
                    skip_song_heat = np.array(list(
                        map(lambda seq: [dqn.song_heat_dict.setdefault(s, default_song_heat) for s in seq],
                            next_train_batch["pre_skip_seq"])))

                    listen_song_feature = np.array(list(
                        map(lambda seq: [dqn.song_feature_dict.setdefault(s, default_song_feature) for s in seq],
                            next_train_batch["pre_play_end_seq"])))
                    listen_song_style = np.array(list(
                        map(lambda seq: [dqn.song_style_dict.setdefault(s, default_song_style) for s in seq],
                            next_train_batch["pre_play_end_seq"])))
                    listen_song_era = np.array(list(
                        map(lambda seq: [dqn.song_era_dict.setdefault(s, default_song_era) for s in seq],
                            next_train_batch["pre_play_end_seq"])))
                    listen_song_lan = np.array(list(
                        map(lambda seq: [dqn.song_lan_dict.setdefault(s, default_song_lan) for s in seq],
                            next_train_batch["pre_play_end_seq"])))
                    listen_song_heat = np.array(list(
                        map(lambda seq: [dqn.song_heat_dict.setdefault(s, default_song_heat) for s in seq],
                            next_train_batch["pre_play_end_seq"])))

                    sub_song_feature = np.array(list(
                        map(lambda seq: [dqn.song_feature_dict.setdefault(s, default_song_feature) for s in seq],
                            next_train_batch["pre_red_seq"])))
                    sub_song_style = np.array(list(
                        map(lambda seq: [dqn.song_style_dict.setdefault(s, default_song_style) for s in seq],
                            next_train_batch["pre_red_seq"])))
                    sub_song_era = np.array(list(
                        map(lambda seq: [dqn.song_era_dict.setdefault(s, default_song_era) for s in seq],
                            next_train_batch["pre_red_seq"])))
                    sub_song_lan = np.array(list(
                        map(lambda seq: [dqn.song_lan_dict.setdefault(s, default_song_lan) for s in seq],
                            next_train_batch["pre_red_seq"])))
                    sub_song_heat = np.array(list(
                        map(lambda seq: [dqn.song_heat_dict.setdefault(s, default_song_heat) for s in seq],
                            next_train_batch["pre_red_seq"])))


                    trash_song_feature = np.array(list(
                        map(lambda seq: [dqn.song_feature_dict.setdefault(s, default_song_feature) for s in seq],
                            next_train_batch["pre_trash_seq"])))
                    trash_song_style = np.array(list(
                        map(lambda seq: [dqn.song_style_dict.setdefault(s, default_song_style) for s in seq],
                            next_train_batch["pre_trash_seq"])))
                    trash_song_era = np.array(list(
                        map(lambda seq: [dqn.song_era_dict.setdefault(s, default_song_era) for s in seq],
                            next_train_batch["pre_trash_seq"])))
                    trash_song_lan = np.array(list(
                        map(lambda seq: [dqn.song_lan_dict.setdefault(s, default_song_lan) for s in seq],
                            next_train_batch["pre_trash_seq"])))
                    trash_song_heat = np.array(list(
                        map(lambda seq: [dqn.song_heat_dict.setdefault(s, default_song_heat) for s in seq],
                            next_train_batch["pre_trash_seq"])))



                    song_feature = np.array(list(
                        map(lambda seq: [dqn.song_feature_dict.setdefault(s, default_song_feature) for s in seq],
                            next_train_batch["song_id"])))
                    song_style = np.array(list(
                        map(lambda seq: [dqn.song_style_dict.setdefault(s, default_song_style) for s in seq],
                            next_train_batch["song_id"])))
                    song_era = np.array(list(
                        map(lambda seq: [dqn.song_era_dict.setdefault(s, default_song_era) for s in seq],
                            next_train_batch["song_id"])))
                    song_lan = np.array(list(
                        map(lambda seq: [dqn.song_lan_dict.setdefault(s, default_song_lan) for s in seq],
                            next_train_batch["song_id"])))
                    song_heat = np.array(list(
                        map(lambda seq: [dqn.song_heat_dict.setdefault(s, default_song_heat) for s in seq],
                            next_train_batch["song_id"])))

                    cur_step, train_op, loss = dqn.get_grident(next_train_batch, song_feature, song_style, song_era, song_lan,
                                                 song_heat,
                                                 skip_song_feature, skip_song_style, skip_song_era, skip_song_lan,
                                                 skip_song_heat,
                                                 listen_song_feature, listen_song_style, listen_song_era,
                                                 listen_song_lan, listen_song_heat,
                                                 sub_song_feature, sub_song_style, sub_song_era, sub_song_lan,
                                                 sub_song_heat,
                                                 trash_song_feature, trash_song_style, trash_song_era, trash_song_lan,
                                                 trash_song_heat, np.array(max_q_batch)
                                                 )


                    batch_loss += loss
                    print ("batch time")
                    print("epoch: {:d} iteration {:d} train_loss: {:6f}".format(epoch, iteration, batch_loss))
                    endtime = datetime.datetime.now()
                    print(endtime - starttime).seconds
                    starttime = endtime
                    iteration += 1
                    batch_loss = 0




                    if iteration % 100 == 0:
                        dqn.update()
                        print("epoch: {:d} iteration {:d} train_loss: {:6f}".format(epoch, iteration, batch_loss))

                    if iteration%1000==0:
                        saver.save(sess, dqn.config.model_path, iteration)





                except tf.errors.OutOfRangeError:
                    break
