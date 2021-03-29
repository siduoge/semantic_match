#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: model_base.py
Author: DST(DST@baidu.com)
Date: 2020/12/22 19:09:31
"""
import tensorflow as tf
import numpy as np


def rcnn_layer(embed, pool_size, out_channel, random_base, l2_reg, active_func=tf.nn.relu, scope_name="cnn"):
    embed = tf.expand_dims(embed, axis=-1)
    hidden_size = embed.get_shape()[2]
    with tf.variable_scope(scope_name+"_kernel_1",reuse=tf.AUTO_REUSE) as scope:
        filter_shape = [1, hidden_size, 1, out_channel]
        conv_w = tf.get_variable(name='conv_w',
                                         shape=filter_shape,
                                         initializer=tf.random_uniform_initializer(-random_base, random_base),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        conv_b = tf.get_variable(name='conv_b',
                                         shape=[out_channels],
                                         initializer=tf.random_uniform_initializer(-random_base, random_base),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        embed_conv = active_func(tf.nn.bias_add(tf.nn.conv2d(embed, conv_w, [1, 1, 1, 1], padding="VALID"), conv_b)) #[b, seq_len+w-1, 1, out_channel]


def atten_matrix(query, doc): #[b, seq_len, 1, h], [b, 1, seq_len, h]
    atten_norm = tf.norm(query - doc, ord=1,axis=-1) # [query_len, doc_len]
    atten_matrix = 1.0/(1.0+atten_norm) #[b, seq_len, seq_len]
    return atten_matrix

def abcnn_pad(embed, pool_size): #[b, seq_len, hidden, 2]
    ret = tf.pad(embed, [[0,0],[pool_size-1, pool_size-1], [0,0], [0,0]], mode="CONSTANT")
    return ret

def abcnn_layer(query, doc, pool_size, max_len, in_channel, out_channels, random_base, l2_reg, active_func=tf.nn.tanh, scope_name="cnn"):
    query_pad = abcnn_pad(query, pool_size) #[b, seq_len+2w-2, h, 2]
    doc_pad = abcnn_pad(doc, pool_size) #[b, seq_len+2w-2, h, 2]
    hidden_size = query.get_shape()[2]
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
        filter_shape = [pool_size, hidden_size, in_channel, out_channels]
        conv_w = tf.get_variable(name='conv_w',
                                         shape=filter_shape,
                                         initializer=tf.random_uniform_initializer(-random_base, random_base),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        conv_b = tf.get_variable(name='conv_b',
                                         shape=[out_channels],
                                         initializer=tf.random_uniform_initializer(-random_base, random_base),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

        query_conv = active_func(tf.nn.bias_add(tf.nn.conv2d(query_pad, conv_w, [1, 1, 1, 1], padding="VALID"), conv_b)) #[b, seq_len+w-1, 1, out_channel]
        doc_conv = active_func(tf.nn.bias_add(tf.nn.conv2d(doc_pad, conv_w, [1, 1, 1, 1], padding="VALID"), conv_b)) #[b, seq_len+w-1, 1, out_channel]
        trans_doc_conv = tf.transpose(doc_conv, [0, 2, 1, 3], name="conv_trans") #[b, 1, seq_len+w-1, hidden]
    # ABCNN2 atten_matrix
    get_atten_matrix = atten_matrix(query_conv, trans_doc_conv) #[b, s_l+w-1, s_l+w-1]
    query_atten = tf.reduce_sum(get_atten_matrix, axis=2) #[b, s_l+w-1]
    doc_atten = tf.reduce_sum(get_atten_matrix, axis=1) #[b, s_l+w-1]
    # pool
    query_wide_pool = wide_pool(query_conv, query_atten, pool_size, max_len)
    doc_wide_pool = wide_pool(doc_conv, query_atten, pool_size, max_len)
    query_all_pool = all_pool(query_conv, pool_size, max_len, out_channels)
    doc_all_pool = all_pool(doc_conv, pool_size, max_len, out_channels)
    return query_wide_pool, doc_wide_pool, query_all_pool, doc_all_pool

def wide_pool(conv, atten, pool_size, seq_len):
    # pool_size
    attention = tf.expand_dims(tf.expand_dims(atten, axis=-1), axis=-1) #[b, seq_len+w-1, 1,1]
    pool_output = []
    for i in range(seq_len):
        pool_output.append(tf.reduce_sum(conv[:, i:pool_size+i,:,:]*attention[:, i:pool_size+i,:,:], axis=1, keep_dims=True))
    ret = tf.concat(pool_output, axis=1)
    return ret

def all_pool(conv, pool_size, seq_len, out_channels):
    pooled = tf.nn.avg_pool(conv, ksize=[1, seq_len+pool_size-1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
    ret = tf.reshape(pooled, [-1, out_channels])
    return ret #[b, out_cha]

def cnn_layer_mask(input_vec, mask_matrix, filters, pools, out_channels, random_base, l2_reg, active_func=tf.nn.tanh, scope_name="cnn"):
    batch_size = tf.shape(input_vec)[0]
    input_channel = 1
    INF_MAX = 1.0e12
    mask_matrix = tf.tile(mask_matrix, [1,1,1,out_channels[0]])
    #input_vec = tf.Print(input_vec, ["input_vec", input_vec[0]], summarize=1000)
    if len(input_vec.get_shape()) == 3:
        input_vec = tf.expand_dims(input_vec, -1)

    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
        for i, filter_size in enumerate(filters):
            filter_shape = [filter_size[0], filter_size[1], input_channel, out_channels[i]]
            conv_w = tf.get_variable(name='conv_w_'+str(i),
                                         shape=filter_shape,
                                         initializer=tf.random_uniform_initializer(-random_base, random_base),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            conv_b = tf.get_variable(name='conv_b_'+str(i),
                                         shape=[out_channels[i]],
                                         initializer=tf.random_uniform_initializer(-random_base, random_base),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

            conv = active_func(tf.nn.bias_add(tf.nn.conv2d(input_vec, conv_w, [1, 1, 1, 1], padding="SAME"), conv_b))
            inf_pad = tf.ones(tf.shape(mask_matrix)) * INF_MAX
            #mask_matrix = tf.Print(mask_matrix, ["mask_matrix", mask_matrix[0]], summarize=1000)
            conv = tf.where(tf.equal(mask_matrix, 1), conv, -inf_pad)
            #conv = tf.Print(conv, ["conv", conv[0]], summarize=1000)
            input_vec = tf.nn.max_pool(conv, ksize=[1, pools[i][0], pools[i][1], 1], strides=[1, 1, 1, 1], padding='VALID')
            zeros = tf.zeros(tf.shape(input_vec))
            ones = tf.ones(tf.shape(input_vec))
            #input_vec = tf.Print(input_vec, ["input_vec", input_vec[0]], summarize=1000)
            mask_matrix = tf.where(tf.equal(input_vec, -INF_MAX), zeros, ones)
            #mask_matrix = tf.Print(mask_matrix, ["mask_matrix", mask_matrix[0]], summarize=1000)
            input_vec = tf.where(tf.equal(mask_matrix, 1), input_vec, zeros)
            #input_vec = tf.Print(input_vec, ["input_vec", input_vec[0]], summarize=1000)
            input_channel = out_channels[i]
    ret = tf.reshape(input_vec, [-1, 26*26*32])
    return ret
    




def cnn_layer(input_vec,  filters, out_channles, hidden_dim, random_base, l2_reg, active_func=tf.nn.tanh, scope_name="cnn"): # relu make 0
    seq_len = input_vec.get_shape()[1]
    hidden_size = input_vec.get_shape()[2]
    if len(input_vec.get_shape()) == 3:
        input_vec = tf.expand_dims(input_vec, -1)
    cnn_features = []
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
        for i, filter_size in enumerate(filters):
            with tf.name_scope("conv_max_pool_"+str(i)):
                filter_shape = [filter_size, hidden_size, 1, out_channles]
                conv_w = tf.get_variable(name='conv_w_'+str(i), 
                                         shape=filter_shape,
                                         initializer=tf.random_uniform_initializer(-random_base, random_base),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
                conv_b = tf.get_variable(name='conv_b_'+str(i),
                                         shape=[out_channles],
                                         initializer=tf.random_uniform_initializer(-random_base, random_base),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
                
                conv = active_func(tf.nn.bias_add(tf.nn.conv2d(input_vec, conv_w, [1, 1, 1, 1], padding="VALID"), conv_b))
                #conv = tf.Print(conv, ["conv", tf.shape(conv)], summarize=1000)
                pooled = tf.nn.max_pool(conv, ksize=[1, seq_len-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
                #pooled = tf.Print(pooled, ["pooled", tf.shape(pooled)], summarize=1000)
                cnn_features.append(pooled)
        cnn_output = tf.concat(cnn_features, axis=-1)
        #cnn_output = tf.Print(cnn_output, ["cnn_output", tf.shape(cnn_output)], summarize=1000)
        flat_shape = len(filters) * out_channles
        cnn_output_flat = tf.reshape(cnn_output, [-1, flat_shape])
        score = tf.layers.dense(cnn_output_flat, hidden_dim, activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    return score

def bilstm_encode(input_embed, seq_len, cell_fw, cell_bw, scope_name):
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
        (query_output_fw_seq, query_output_bw_seq), _1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,                                                        cell_bw=cell_bw,
                 inputs=input_embed,                                                     sequence_length=seq_len,
                 dtype=tf.float32,
                 scope=scope_name)
        ret = tf.concat([query_output_fw_seq, query_output_bw_seq], axis=-1)
        return ret

def bilstm_encode_final(input_embed, seq_len, cell_fw, cell_bw, scope_name):
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
        _1, (query_output_fw_seq, query_output_bw_seq)= tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,

      cell_bw=cell_bw,

      inputs=input_embed,

      sequence_length=seq_len,

      dtype=tf.float32,

      scope=scope_name)
        ret = tf.concat([tf.concat(query_output_fw_seq,axis=-1), tf.concat(query_output_bw_seq, axis=-1)], axis=-1)
        return ret

def matrix_mask(query_len, doc_len, max_len):
    query_sequence_mask = tf.sequence_mask(query_len, max_len, name="query_mask")
    doc_sequence_mask = tf.sequence_mask(doc_len, max_len, name="doc_mask")
    tile_doc_mask =  tf.cast(tf.tile(tf.expand_dims(doc_sequence_mask, axis=1), [1, max_len, 1]) , dtype=tf.float32)#[b, seq_len ,seq_len]
    tile_query_mask = tf.cast(tf.tile(tf.expand_dims(doc_sequence_mask, axis=2), [1, 1, max_len]),  dtype=tf.float32) #[b, seq_len ,seq_len]
    mask_matrix = tile_doc_mask * tile_query_mask
    return mask_matrix

def max_and_avg(embed):
    seq_len = embed.get_shape()[1]
    hidden_dim = tf.shape(embed)[-1]
    max_pool = tf.reduce_sum(embed, axis=1)
    avg_pool = tf.reduce_mean(embed, axis=1)
    max_pool_output = tf.reshape(max_pool, [-1, hidden_dim])
    avg_pool_output = tf.reshape(avg_pool, [-1, hidden_dim])
    ret = tf.concat([max_pool_output, avg_pool_output], axis=-1)
    return ret
