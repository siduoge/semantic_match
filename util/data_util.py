#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: data_util.py
Author: DST(DST@baidu.com)
Date: 2020/12/23 09:57:05
"""
import sys
import numpy as np
from util.config import *
import random

def loadVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')

    vocab = {}
    rev_vocab = {}
    with open(path) as fd:
        for line in fd:
            line = line.rstrip('\r\n')
            line = line.split(" ")
            index = int(line[1])+2
            vocab[line[0]] = index
        vocab["_pad"] = 0
        vocab["_unk"] = 1
    rev_vocab = dict([(index, cha) for (cha, index) in vocab.items()])
    return vocab, rev_vocab

def load_all_data(input_path, max_len):
    paper = {}
    paper_ids = []
    paper_len = {}
    with open(input_path, "r") as fr:
        fr.readline()
        for line in fr:
            data = line.strip().split("\t")
            paper_id = data[0]
            paper_ids.append(paper_id)
            word_list  = np.zeros([1, max_len]) # [ 0 for i in range(max_len)]
            file_list = data[1].split(" ")
            paper_len[paper_id] =len(file_list)
            file_list = file_list[:max_len]
            for i, val in enumerate(file_list):
                word_list[0][i] = int(val) + 2
            paper[paper_id]  = word_list[:]
    return paper, paper_len, paper_ids

def load_relation(input_file):
    fr = open(input_file, "r")
    _ = fr.readline()
    dic = {}
    for line in fr:
        label, paper1, paper2 = line.strip().split(" ")
        if paper1 not in dic:
            dic[paper1] = [[], []]
        if paper2 not in dic:
            dic[paper2] = [[], []]
        if int(label) == 1:
            dic[paper1][0].append(paper2)
            dic[paper2][0].append(paper1)
        else:
            dic[paper1][1].append(paper2)
            dic[paper2][1].append(paper1)            
        
    return dic



def load_data_dssm(papers, papers_len, input_file):
    fr = open(input_file, "r")
    _ = fr.readline()
    querys = []
    docs = []
    labels = []
    querys_len = []
    docs_len = []
    for line in fr:
        query_vec = []
        doc_vecs = []
        label, query_index, doc_index = line.strip().split(" ")
        labels.append(int(label))
        query_vec = papers[query_index]
        doc_vec = papers[doc_index]
        querys.append(query_vec)
        docs.append(doc_vec)
        querys_len.append(papers_len[query_index])
        docs_len.append(papers_len[doc_index])
    np_querys = np.concatenate(querys, axis=0)
    np_docs = np.concatenate(docs, axis=0)
    np_labels = np.array(labels)
    fr.close()
    return np_querys, np_docs, np_labels, np.array(querys_len), np.array(docs_len)

def load_data_dssm_train(papers, papers_len, paper_relation, paper_ids, input_file, num_negs):
    # 每个正样例，找63个负样例
    fr = open(input_file, "r")
    _ = fr.readline()
    querys = []
    docs = []
    labels = []
    querys_len = []
    docs_len = []
    max_paper_len = len(paper_ids)
    for line in fr:
        query_vec = []
        doc_vecs = []
        label, query_index, doc_index = line.strip().split(" ")
        neg_list = []
        doc_lens = []
        if int(label) == 1:
            labels.append(0)
            query_vec = papers[query_index]
            querys_len.append(papers_len[query_index])
            doc_vec = papers[doc_index]
            doc_vecs.append(doc_vec)
            doc_lens.append(papers_len[doc_index])
            pos_list = paper_relation[query_index][0] + paper_relation[doc_index][0]
            neg_list = paper_relation[query_index][1] + paper_relation[doc_index][1]

            ret_num = 0
            if len(neg_list) < num_negs:
                ret_num = num_negs - len(neg_list)
            
            for i in range(ret_num):
                while True:
                    index = random.randint(0, max_paper_len-1)
                    paper_id = paper_ids[index]
                    if paper_id not in pos_list and paper_id not in neg_list:
                        neg_list.append(paper_id) 
                        break
            for val in neg_list[:num_negs]:
                doc_vecs.append(papers[val])
                doc_lens.append(papers_len[val])
            querys.append(query_vec)
            doc_vec_vec = np.expand_dims(np.concatenate(doc_vecs, axis=0), axis=0)
            #print(doc_vec_vec.shape)
            docs.append(doc_vec_vec)
            docs_len.append(doc_lens)
    np_querys = np.concatenate(querys, axis=0)
    np_docs = np.concatenate(docs, axis=0)
    np_labels = np.array(labels)
    np_query_len = np.array(querys_len)
    np_doc_len = np.array(docs_len)
    fr.close()
    return np_querys, np_docs, np_labels, np_query_len, np_doc_len

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    #shuffle = False
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

                    
            
                
                
 
     
          
    
