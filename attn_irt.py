import tensorflow as tf
import numpy as np
from official.transformer.v2 import embedding_layer 
from official.nlp.modeling import layers
from official.nlp.transformer import model_utils #might be official.nlp.transformer.model_utils
from dkt_irt import q_constraint, q_1pl_constraint
import tensorflow_probability as tfp

class ATTN_IRT(tf.keras.Model):
    def __init__(self, params):
        super(ATTN_IRT, self).__init__()
        self.params = params
        self.q_matrix = self.params['q_matrix']
        if params['1pl']:
          weight_constraint = q_1pl_constraint(self.q_matrix)
        else:
          weight_constraint = q_constraint(self.q_matrix)
        self.embedding_layer = tf.keras.layers.Embedding(params['encoder_vocab_size'], params['hidden_size'])
        if params['pos_enc']:
          self.pos_encoding = np.zeros((params['max_len'], params['hidden_size']))
          sin_cos = model_utils.get_position_encoding(self.params['max_len'], self.params['hidden_size'])
          self.pos_encoding[:, 0:params['hidden_size']:2] = sin_cos[:, :params['hidden_size']//2]
          self.pos_encoding[:, 1:params['hidden_size']:2] = sin_cos[:, params['hidden_size']//2:]
        self.attention_layer = SelfAttention(params['hidden_size'], num_heads=params['num_heads'], dropout=params['dropout'], Q=params['q_matrix'], q_mask_attn=params['q_mask_attn'])
        if params['layer_norm']:
        	self.layer_norm = tf.keras.layers.LayerNormalization() 
        self.hidden_layer = tf.keras.layers.Dense(2 * params['num_skills'], activation=params['hid_activation'], name='hidden_layer')
        self.skill_layer = tf.keras.layers.Dense(params['num_skills'], activation='linear', name='skill_layer')

        self.problem_layer = tf.keras.layers.Dense(params['num_items'], activation='sigmoid',
                                                   kernel_constraint = weight_constraint, name='problem_layer')
        # need to mask out future interactions
        z = int(self.params['max_len'] * (self.params['max_len'] + 1) / 2)
        ones = np.ones(z, dtype='float32')
        tri = tfp.math.fill_triangular(ones) # lower tri, all 1s
        future_mask = 1 - tf.exp(1/tri - 1) # lower tri is all 0s, top tri is all -inf
        self.future_mask = tf.reshape(future_mask, (1,1, self.params['max_len'], self.params['max_len']))
    
    def get_config(self):
        return {'params': self.params}
    
    def get_skills(self, prob_x):
        embedded_x = self.embedding_layer(prob_x)
        if self.params['pos_enc']:
	        embedded_x += self.pos_encoding
        cor, attn = self.get_attn(prob_x)
        add_norm = self.layer_norm(embedded_x + attn)
        h = self.hidden_layer(add_norm) 
        skills_variance = self.skill_layer(h)
        return skills_variance
        
    def get_attn(self, prob_x):
        batch_size = tf.shape(prob_x)[0]
        embedded_x = self.embedding_layer(prob_x)
        prob_id = tf.cast((prob_x - 4) // 2, dtype=tf.int32)
        if self.params['pos_enc']:
	        embedded_x += self.pos_encoding
        mask = tf.tile(self.future_mask, [batch_size, 1, 1, 1]) # mask has shape [batch, 1, seq, seq]
        cor, attn = self.attention_layer(embedded_x, mask, prob_id, training=False)
        return [cor, attn]
    
    def call(self, inputs, training=False):
        prob_x, prob_y2 = inputs
        
        next_prob_id = tf.cast((prob_y2 - 4) // 2, dtype=tf.int32)
        prob_id = tf.cast((prob_x - 4) // 2, dtype=tf.int32)
        batch_size = tf.shape(prob_x)[0]
        sequence_length = tf.shape(prob_x)[1]
        
        with tf.name_scope('ATTN_IRT'):
            embedded_x = self.embedding_layer(prob_x)
            if self.params['pos_enc']:
	            embedded_x += self.pos_encoding
            mask = tf.tile(self.future_mask, [batch_size, 1, 1, 1]) # mask has shape [batch, 1, seq, seq]
            
            weights, attention = self.attention_layer(embedded_x, mask, prob_id, training)
            
            if self.params['layer_norm']:
            	attention = self.layer_norm(embedded_x + attention) 
            
            hid = self.hidden_layer(attention)
            skills = self.skill_layer(hid)
            problems = self.problem_layer(skills)
            mask = tf.one_hot(next_prob_id, depth=self.params['num_items'])
            output = tf.einsum('abi,abi->ab', problems, mask)
            return output      
      
class SelfAttention(tf.keras.layers.Layer):
  # based on https://github.com/tensorflow/models/blob/master/official/nlp/transformer/attention_layer.py
  def __init__(self, hidden_size, num_heads, dropout, Q, q_mask_attn=False):
    super(SelfAttention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.dropout = dropout

    # next few lines for the q_mask_attn feature -- which isn't a fully flushed out idea yet and doesn't work great
    items_items = np.matmul(np.transpose(Q), Q)
    items_items = -1/items_items
    self.items_items = items_items * (items_items <= -np.inf) # entry = -inf if items don't share a skill, 0 if they do share a skill
    self.q_mask_attn = q_mask_attn
    
    size_per_head = self.hidden_size // self.num_heads
    self.query_dense_layer = layers.DenseEinsum(
      output_shape=(self.num_heads, size_per_head),
      use_bias=False,
      name='query')
    self.key_dense_layer = layers.DenseEinsum(
      output_shape=(self.num_heads, size_per_head),
      use_bias=False,
      name='key') 
    self.value_dense_layer = layers.DenseEinsum(
      output_shape=(self.num_heads, size_per_head),
      use_bias=False,
      name='value') 
    self.output_dense_layer = layers.DenseEinsum(
      output_shape=self.hidden_size,
      num_summed_dimensions=2,
      use_bias=False,
      name='output_transform')
    
  def call(self, x, bias, prob_id, training):
    # x is shape [batch_size, seq_len, hidden_size]
    # we can use bias input to mask out future attentions 
    # bias is shape [bath_size, 1, seq_len, length_query]
    
    batch_size = x.shape[0]
    max_len = x.shape[1]
    
    query = self.query_dense_layer(x)
    key = self.key_dense_layer(x)
    value = self.value_dense_layer(x)
    
    depth = self.hidden_size // self.num_heads
    query *= depth ** -0.5
	
    logits = tf.einsum('BTNH,BFNH->BNFT', key, query)
    # could try symmetric attention (only keys, no query)
    #logits = tf.einsum('BTNH,BFNH->BNFT', key, key)
    logits += bias # the bias passed in here is a future mask
    
    # mutliply logits by -inf if items don't share a skill (from Q-matrix)
    # this doesn't work quite like I want it to
    if self.q_mask_attn:
      ind = prob_id[:,1:, tf.newaxis] # the first entry in prob_id corresponds to start token
      out = tf.gather_nd(self.items_items, ind)
      out = tf.transpose(out, (0,2,1))
      out = tf.gather(out, prob_id[:,1:], axis=1, batch_dims=1)
      out = tf.transpose(out, (0,2,1))
      paddings = tf.constant([[0,0],[1,0],[1,0]]) # pad on top and left
      common_skills_mask = tf.cast(tf.pad(out, paddings, constant_values=-2**32), dtype=tf.float32)
      common_skills_mask = tf.expand_dims(common_skills_mask, axis=1)
      logits += common_skills_mask
    
    weights = tf.nn.softmax(logits, name='attention_weights')
    if training:
      weights = tf.nn.dropout(weights, rate=self.dropout)
    attention_output = tf.einsum('BNFT,BTNH->BFNH', weights, value)
    
    attention_output = self.output_dense_layer(attention_output)
    return [weights, attention_output]
  
  
