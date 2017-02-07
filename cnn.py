#coding=utf-8
"""
该类通过CNN计算每个句子的特征，然后通过计算问题和相似问题，以及不相似问题的间隔，用于解决qa，目标是使得间隔最大。
https://arxiv.org/abs/1508.01585
"""
import tensorflow as tf

class CNN_LSTM(object):
    def __init__(self, batch_size, sequence_len, word_embedding, embedding_size, filter_sizes, num_filters, rnn_size, num_rnn_layers, l2_reg_lambda=0.0):
        """
        sequence_len: the max length of sentence
        word_embedding: word_embedding
        embedding_size: the dim of embedding
        filter_size: the size of filter, eg:[1,2,3,4,5]
        num_filters: the number of filter, how many filter contain in every layer, eg:128  
        l2_reg_lambda: l2_reg_lambda is use to limit overfit
        """
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        # 定义输入变量
        self.org_quest = tf.placeholder(tf.int32, [None, sequence_len], name = "ori_quest")
        self.cand_quest = tf.placeholder(tf.int32, [None, sequence_len], name="cand_quest")
        self.neg_quest = tf.placeholder(tf.int32, [None, sequence_len], name="negative_quest")
        self.keep_dropout = tf.placeholder(tf.float32, name="dropout")

        # 定义词向量
        with tf.name_scope("embedding"):
            W = tf.Variable(tf.to_float(word_embedding), trainable=True, name="W")

            self.org_quest_embedding = tf.nn.embedding_lookup(W, self.org_quest)
            self.cand_quest_embedding = tf.nn.embedding_lookup(W, self.cand_quest)
            self.neg_quest_embedding = tf.nn.embedding_lookup(W, self.neg_quest)


        #build LSTM network
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, forget_bias=0.0, state_is_tuple=True)
        lstm_cell =  tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob = self.keep_dropout
            )
        self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_rnn_layers, state_is_tuple=True)
        self._initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)

        # lstm (seq_len, batch_size, rnn_size)
        ori_out, cand_out, neg_out = self.cal_rnn(self.cell, self._initial_state, self.sequence_len, self.org_quest_embedding, self.cand_quest_embedding, self.neg_quest_embedding, "rnn_scope")
        ori_out_put, cand_out_put, neg_out_put = tf.transpose(ori_out, perm=[1,0,2]), tf.transpose(cand_out, perm=[1,0,2]), tf.transpose(neg_out, perm=[1,0,2])
        # 扩充词向量为卷积对应的格式，即(batch_size, sequence_len, embedding_size, in_channels)
        ori_out_put = tf.expand_dims(ori_out_put, -1, name="ori")
        cand_out_put = tf.expand_dims(cand_out_put, -1, name="cand")
        neg_out_put = tf.expand_dims(neg_out_put, -1, name="neg")
        #outputs_ori, outputs_cand, outputs_neg = self.max_pool(ori_out, cand_out, neg_out)#(batch_size, rnn_size)
        # 对输入变量进行卷积
        ori_pooled_outputs = []
        cand_pooled_outputs = []
        neg_pooled_outputs = []
        for idx, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s"%   filter_size):
                # filter shape is (weight, width, in_channels, out_channnels)
                filter_shape = [filter_size, self.rnn_size, 1, num_filters]
                filter_weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_weight")
                filter_bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="filter_bias")
                
                # convolution (batch_size, sequence_len - filter_size + 1, in_channels, out_channnels)
                conv_ori = tf.nn.conv2d(ori_out_put, filter_weight, strides=[1,1,1,1], padding="VALID", name="conv_ori")

                # apply nonlinearity
                relu_output_ori = tf.nn.relu(tf.nn.bias_add(conv_ori, filter_bias), name="relu_ori")

                # Maxpooling over the outputs
                ori_pooled = tf.nn.max_pool(
                    relu_output_ori,
                    ksize=[1, self.sequence_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                ori_pooled_outputs.append(ori_pooled)

                # convolution (batch_size, sequence_len - filter_size + 1, in_channels, out_channnels)
                conv_cand = tf.nn.conv2d(cand_out_put, filter_weight, strides=[1,1,1,1], padding="VALID", name="conv_cand")

                # apply nonlinearity
                relu_output_cand = tf.nn.relu(tf.nn.bias_add(conv_cand, filter_bias), name="relu_cand")

                # Maxpooling over the outputs
                cand_pooled = tf.nn.max_pool(
                    relu_output_cand,
                    ksize=[1, self.sequence_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                cand_pooled_outputs.append(cand_pooled)

                # convolution (batch_size, sequence_len - filter_size + 1, in_channels, out_channnels)
                conv_neg = tf.nn.conv2d(neg_out_put, filter_weight, strides=[1,1,1,1], padding="VALID", name="conv_neg")

                # apply nonlinearity
                relu_output_neg = tf.nn.relu(tf.nn.bias_add(conv_neg, filter_bias), name="relu_neg")

                # Maxpooling over the outputs
                neg_pooled = tf.nn.max_pool(
                    relu_output_neg,
                    ksize=[1, self.sequence_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                neg_pooled_outputs.append(neg_pooled)

        # concate output of each filter
        cnn_ori_output = tf.squeeze(tf.concat(3, ori_pooled_outputs), [1, 2], name="cnn_ori_output")
        cnn_cand_output = tf.squeeze(tf.concat(3, cand_pooled_outputs), [1, 2], name="cnn_cand_output")
        cnn_neg_output = tf.squeeze(tf.concat(3, neg_pooled_outputs), [1, 2], name="cnn_neg_output")


        # reshape out 
        out_ori = cnn_ori_output 
        out_cand = cnn_cand_output 
        out_neg = cnn_neg_output 

        # dropout
        out_ori = tf.nn.dropout(out_ori, self.keep_dropout)
        out_cand = tf.nn.dropout(out_cand, self.keep_dropout)
        out_neg = tf.nn.dropout(out_neg, self.keep_dropout)

        # cal cosine simulation
        ori_seq_len = tf.sqrt(tf.reduce_sum(tf.mul(out_ori, out_ori), 1), name="sqrt_ori")
        cand_seq_len = tf.sqrt(tf.reduce_sum(tf.mul(out_cand, out_cand), 1), name="sqrt_cand")
        neg_seq_len = tf.sqrt(tf.reduce_sum(tf.mul(out_neg, out_neg), 1), name="sqrt_neg")

        ori_cand_dist = tf.reduce_sum(tf.mul(out_ori, out_cand), 1, name="ori_cand")
        ori_neg_dist = tf.reduce_sum(tf.mul(out_ori, out_neg), 1, name="ori_neg")

        # cal the score
        with tf.name_scope("score"):
            self.ori_cand_score = tf.div(ori_cand_dist, tf.mul(ori_seq_len, cand_seq_len), name="score_positive")
            self.ori_neg_score = tf.div(ori_neg_dist, tf.mul(ori_seq_len, neg_seq_len), name="score_negative")

        # the target function 
        zero = tf.fill(tf.shape(self.ori_cand_score), 0.0)
        margin = tf.fill(tf.shape(self.ori_cand_score), 0.05)
        l2_loss = tf.constant(0.0)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(zero, tf.sub(margin, tf.sub(self.ori_cand_score, self.ori_neg_score)))
            self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
        
        # cal accurancy
        with tf.name_scope("acc"):
            self.correct = tf.equal(zero, self.losses)
            self.acc = tf.reduce_mean(tf.cast(self.correct, "float"), name="acc")

    def max_pool(self, ori_out_put, cand_out_put, neg_out_put):
	# ori_out_put(num_unroll_steps * batch_size * rnn_size) 
        with tf.name_scope("regulation_layer"):
            ori_out_put, cand_out_put, neg_out_put = tf.transpose(ori_out_put, perm=[1,2,0]), tf.transpose(cand_out_put, perm=[1,2,0]), tf.transpose(neg_out_put, perm=[1,2,0])
            ori_batch_output, cand_batch_output, neg_batch_output = [], [], []
            for sent_idx in range(self.batch_size):
	            ori_batch_output.append(tf.reduce_max(ori_out_put[sent_idx], 1))
	            cand_batch_output.append(tf.reduce_max(cand_out_put[sent_idx], 1))
	            neg_batch_output.append(tf.reduce_max(neg_out_put[sent_idx], 1))
        self.out_ori = tf.nn.tanh(ori_batch_output, name="tanh_ori")#(batch_size, 2 * rnn_size)
        self.out_cand = tf.nn.tanh(cand_batch_output, name="tanh_cand")
        self.out_neg = tf.nn.tanh(neg_batch_output, name="tanh_neg")
        return self.out_ori, self.out_cand, self.out_neg

    def cal_rnn(self, cell, _initial_state, num_unroll_steps, ori_quests, cand_quests, neg_quests, scope):
        ori_out_put=[]
        cand_out_put=[]
        neg_out_put=[]
        ori_state = _initial_state
        cand_state = _initial_state
        neg_state = _initial_state
        with tf.variable_scope(scope):
            for time_step in range(num_unroll_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (ori_cell_output, ori_state)=cell(ori_quests[:,time_step, :], ori_state)
                ori_out_put.append(ori_cell_output)
                
                tf.get_variable_scope().reuse_variables()
                (cand_cell_output, cand_state)=cell(cand_quests[:,time_step, :], cand_state)
                cand_out_put.append(cand_cell_output)

                tf.get_variable_scope().reuse_variables()
                (neg_cell_output, neg_state)=cell(neg_quests[:,time_step, :], neg_state)
                neg_out_put.append(neg_cell_output)
        return ori_out_put, cand_out_put, neg_out_put
