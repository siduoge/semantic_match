#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# 
########################################################################
 
"""
File: esim.py
Author: DST(DST@baidu.com)
Date: 2021/01/01 11:50:13
"""
import tensorflow as tf
from model.model_base import *
from util.config import *
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

class ABCNN(object):
    def __init__(self, FLAGS):
        self.config = FLAGS
        self.filters = eval(self.config.match_filters)
        self.pools = eval(self.config.match_pools)
        self.out_channels = eval(self.config.match_channels)
        self.add_placeholder()
        self.input2embed()
        self.model()

    def add_placeholder(self):
        self.input_query = tf.placeholder(tf.int32, [None, self.config.max_len], name='query_input')
        self.input_doc = tf.placeholder(tf.int32, [None, self.config.max_len], name='doc_input')
        self.query_seq_len = tf.placeholder(tf.int32, [None], name="query_seq_len")
        self.doc_seq_len = tf.placeholder(tf.int32, [None], name="doc_seq_len")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.label = tf.placeholder(tf.int32, [None], name="label")

    def input2embed(self):
        self.query_embedding = tf.get_variable(name='word_embedding',
                                                   shape=[self.config.word_vocab_size, self.config.word_embedding_dim],
                                                   dtype=tf.float32,
                                                   initializer=tf.random_uniform_initializer(maxval=-1., minval=1., seed=0))
        self.query_embed = tf.nn.embedding_lookup(self.query_embedding, self.input_query) #[b, seq_len, h]
        self.doc_embed = tf.nn.embedding_lookup(self.query_embedding, self.input_doc)

    def ABCNN_model(self, scope):
		# ABCNN1 attention matrix
        query = tf.expand_dims(self.query_embed, axis=2) #[b, l,1,h]
        doc = tf.expand_dims(self.query_embed, axis=1) #[b, 1,l,h]
        init_atten_matrix = atten_matrix(query, doc)
        feature_map_query = tf.layers.dense(init_atten_matrix, self.config.word_embedding_dim, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.l2_reg), name="query_weigths")
        feature_map_doc = tf.layers.dense(init_atten_matrix, self.config.word_embedding_dim, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.l2_reg), name="doc_weigths")
        pic_query = tf.concat([tf.expand_dims(self.query_embed, axis=-1), tf.expand_dims(feature_map_query, axis=-1)], axis=-1)
        pic_doc = tf.concat([tf.expand_dims(self.doc_embed, axis=-1), tf.expand_dims(feature_map_doc, axis=-1)], axis=-1) #[b, seq_len, hidden, 2]

        # first layer
        output_features = []
        for i in range(self.config.abcnn_layers):
            if i == 0:
                pic_query, pic_doc, query_ap, doc_ap = abcnn_layer(pic_query, pic_doc, self.config.abcnn_pool_size, self.config.max_len, 2, self.config.abcnn_out_channels, self.config.random_base, self.config.l2_reg, active_func=tf.nn.tanh, scope_name="cnn_layer_" + str(i))
            else:
                pic_query, pic_doc, query_ap, doc_ap = abcnn_layer(pic_query, pic_doc, self.config.abcnn_pool_size, self.config.max_len, self.config.abcnn_out_channels, self.config.abcnn_out_channels, self.config.random_base, self.config.l2_reg, active_func=tf.nn.tanh, scope_name="cnn_layer_" + str(i))
            output_features.append(query_ap)
            output_features.append(doc_ap)
        output = tf.concat(output_features, axis=-1)
        dense_output = tf.layers.dense(output, self.config.abcnn_hidden, activation=tf.nn.relu)
        self.prob = tf.layers.dense(dense_output, self.config.num_class)


    def loss(self):
        with tf.name_scope("loss"):
            label = tf.one_hot(self.label, 2)
            self.origin_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=self.prob)
            self.loss = tf.reduce_mean(self.origin_loss)
        with tf.name_scope("accuarcy"):
            doc_predict = tf.cast(tf.argmax(self.prob, axis=-1, name="predict"), dtype=tf.int32)
            correct_predictions = tf.equal(doc_predict, self.label, name='correct_predictions')
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
    def model(self):
        scope = "ABCNN"
        self.ABCNN_model(scope)
        self.loss()	
