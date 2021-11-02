from typing import Optional

import tensorflow as tf

from docqa.nn.layers import AttentionMapper, MergeLayer, SequenceEncoder, get_keras_initialization, SequenceMapper, \
    Mapper, SequenceMultiEncoder
from docqa.nn.ops import VERY_NEGATIVE_NUMBER, exp_mask
from docqa.nn.similarity_layers import SimilarityFunction, compute_attention_mask

"""
Module for non-recurrent attention layers
"""


class StaticAttention(AttentionMapper):
    """ Basic non-recurrent attention using the given SimilarityFunction """

    def __init__(self, attention: SimilarityFunction, alignment_bias=None):
        self.attention = attention
        self.alignment_bias = alignment_bias

    def apply(self, is_train, x, keys, memories, x_mask=None, mem_mask=None):
        x_word_dim = tf.shape(x)[1]
        key_word_dim = tf.shape(keys)[1]

        # (batch, x_word, key_word)
        dist_matrix = self.attention.get_scores(x, keys)

        joint_mask = compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim)
        dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))

        if self.alignment_bias is None:
            select_probs = tf.nn.softmax(dist_matrix)
        else:
            # Compute softmax with an additional bias term, this allows the model to 'ignore' the memories
            # if needed since the sum of the weights given to each memory can be < 1.
            bias = tf.exp(tf.get_variable("no-alignment-bias", initializer=tf.constant(-1.0, dtype=tf.float32)))
            dist_matrix = tf.exp(dist_matrix)
            select_probs = dist_matrix / (tf.reduce_sum(dist_matrix, axis=2, keep_dims=True) + bias)

        #  Too (batch, x_word, memory_dim)
        response = tf.matmul(select_probs, memories)
        
        return response

    def __setstate__(self, state):
        if "alignment_bias" not in state["state"]:
            state["state"]["alignment_bias"] = None
        super().__setstate__(state)


class StaticAttentionSelf(SequenceMapper):
    """ Basic non-recurrent attention a sequence and itself using the given SimilarityFunction """

    def __init__(self, attention: SimilarityFunction,
                 alignment_bias=True):
        self.alignment_bias = alignment_bias
        self.attention = attention

    def apply(self, is_train, x, x_mask=None):
        x_word_dim = tf.shape(x)[1]

        # (batch, x_word, key_word)
        dist_matrix = self.attention.get_scores(x, x)
        dist_matrix += tf.expand_dims(tf.eye(x_word_dim) * VERY_NEGATIVE_NUMBER, 0)  # Mask out self

        joint_mask = compute_attention_mask(x_mask, x_mask, x_word_dim, x_word_dim)
        if joint_mask is not None:
            dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))

        if not self.alignment_bias:
            select_probs = tf.nn.softmax(dist_matrix)
        else:
            # Allow zero-attention by adding a learned bias to the normalizer
            bias = tf.exp(tf.get_variable("no-alignment-bias", initializer=tf.constant(-1.0, dtype=tf.float32)))
            dist_matrix = tf.exp(dist_matrix)
            select_probs = dist_matrix / (tf.reduce_sum(dist_matrix, axis=2, keep_dims=True) + bias)

        response = tf.matmul(select_probs, x)  # (batch, x_words, q_dim)

        return response

    def __setstate__(self, state):
        if "state" in state:
            state["state"]["alignment_bias"] = True
        super().__setstate__(state)


class NullAttention(AttentionMapper):
    def apply(self, is_train, x, keys, memories, mask=None, memory_mask=None):
        return x


class BiAttention(AttentionMapper):
    """ Bi-Directonal Attention from https://arxiv.org/abs/1611.01603 """

    def __init__(self, sim: SimilarityFunction, q2c: bool, query_dots: bool=True):
        self.sim = sim
        self.q2c = q2c
        self.query_dots = query_dots

    def apply(self, is_train, x, keys, memories, x_mask=None, mem_mask=None):
        x_word_dim = tf.shape(x)[1]
        key_word_dim = tf.shape(keys)[1]

        dist_matrix = self.sim.get_scores(x, keys)
        joint_mask = compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim)
        if joint_mask is not None:
            dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))
        query_probs = tf.nn.softmax(dist_matrix)  # probability of each mem_word per x_word

        # Batch matrix multiplication to get the attended vectors
        select_query = tf.matmul(query_probs, memories)  # (batch, x_words, q_dim)

        if not self.q2c:
            if self.query_dots:
                return tf.concat([x, select_query, x * select_query], axis=2)
            else:
                return tf.concat([x, select_query], axis=2)

        # select query-to-context
        context_dist = tf.reduce_max(dist_matrix, axis=2)  # (batch, x_word``s)
        context_probs = tf.nn.softmax(context_dist)  # (batch, x_words)
        select_context = tf.einsum("ai,aik->ak", context_probs, x)  # (batch, x_dim)
        select_context = tf.expand_dims(select_context, 1)

        if self.query_dots:
            return tf.concat([x, select_query, x * select_query, x * select_context], axis=2)
        else:
            return tf.concat([x, select_query, x * select_context], axis=2)

    def __setstate__(self, state):
        if "state" in state:
            if "query_dots" not in state["state"]:
                state["state"]["query_dots"] = True
        super().__setstate__(state)


class StaticAttentionWithEncoder(AttentionMapper):
    """ BiDaF like layer, except will allow the query vector to come from an arbitrary encoder layer """

    def __init__(self, attention: SimilarityFunction,
                 encoder_layer: SequenceEncoder,
                 alignment_bias=None):
        self.attention = attention
        self.encoder_layer = encoder_layer
        self.alignment_bias = alignment_bias

    def apply(self, is_train, x, keys, memories, x_mask=None, mem_mask=None):
        x_word_dim = tf.shape(x)[1]
        key_word_dim = tf.shape(keys)[1]

        # (batch, x_word, key_word)
        dist_matrix = self.attention.get_scores(x, keys)

        joint_mask = compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim)
        dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))

        if self.alignment_bias is None:
            select_probs = tf.nn.softmax(dist_matrix)
        else:
            bias = tf.exp(tf.get_variable("no-alignment-bias", initializer=tf.constant(-1.0, dtype=tf.float32)))
            dist_matrix = tf.exp(dist_matrix)
            select_probs = dist_matrix / (tf.reduce_sum(dist_matrix, axis=2, keep_dims=True) + bias)

        #  Too (batch, x_word, memory_dim)
        response = tf.matmul(select_probs, memories)

        with tf.variable_scope("encode_keys"):
            encoded = self.encoder_layer.apply(is_train, keys, mem_mask)

        return tf.concat([x, response, x * response, x * tf.expand_dims(encoded, 1)], axis=2)



class AttentionEncoder(SequenceEncoder):
    def __init__(self, key_mapper: SequenceMapper=None,
                 post_process: Mapper=None,
                 init="glorot_uniform"):
        self.init = init
        self.key_mapper = key_mapper
        self.post_process = post_process

    def apply(self, is_train, x, mask=None):
        if self.key_mapper is not None:
            with tf.variable_scope("map_keys"):
                keys = self.key_mapper.apply(is_train, x, mask)
        else:
            keys = x

        weights = tf.get_variable("weights", keys.shape.as_list()[-1], dtype=tf.float32,
                                  initializer=get_keras_initialization(self.init))
        dist = tf.tensordot(keys, weights, axes=[[2], [0]])  # (batch, x_words)
        dist = exp_mask(dist, mask)
        dist = tf.nn.softmax(dist)

        out = tf.einsum("ajk,aj->ak", x, dist)  # (batch, x_dim)

        if self.post_process is not None:
            with tf.variable_scope("post_process"):
                out = self.post_process.apply(is_train, out)
        return out


class MultiAttentionEncoder(SequenceMultiEncoder):
    def __init__(self, n_encodings: int, bias: bool=False, key_mapper: SequenceMapper=None,
                 post_process: Mapper=None,
                 init="glorot_uniform"):
        self.init = init
        self.bias = bias
        self.n_encodings = n_encodings
        self.key_mapper = key_mapper
        self.post_process = post_process

    def apply(self, is_train, x, mask=None):
        if self.key_mapper is not None:
            with tf.variable_scope("map_keys"):
                keys = self.key_mapper.apply(is_train, x, mask)
        else:
            keys = x

        weights = tf.get_variable("weights", (keys.shape.as_list()[-1], self.n_encodings), dtype=tf.float32,
                                  initializer=get_keras_initialization(self.init))
        dist = tf.tensordot(keys, weights, axes=[[2], [0]])  # (batch, x_words, n_encoding)
        if self.bias:
            dist += tf.get_variable("bias", (1, 1, self.n_encodings),
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
        if mask is not None:
            bool_mask = tf.expand_dims(tf.cast(tf.sequence_mask(mask, tf.shape(x)[1]), tf.float32), 2)
            dist = bool_mask * bool_mask + (1 - bool_mask) * VERY_NEGATIVE_NUMBER

        dist = tf.nn.softmax(dist, dim=1)

        out = tf.einsum("ajk,ajn->ank", x, dist)  # (batch, n_encoding, feature)

        if self.post_process is not None:
            with tf.variable_scope("post_process"):
                out = self.post_process.apply(is_train, out)
        return out
