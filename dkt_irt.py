import tensorflow as tf
import numpy as np
from official.transformer.v2 import embedding_layer

class DKT_IRT(tf.keras.Model):
    def __init__(self, params):
        super(DKT_IRT, self).__init__()
        self.params = params
        self.q_matrix = params['q_matrix']
        self.embedding_layer = tf.keras.layers.Embedding(params['encoder_vocab_size'], params['hidden_size'])
        if params['lstm']:
            self.recurrent_layer = tf.keras.layers.LSTM(params['hidden_size'], activation = 'tanh', return_sequences = True)
            self.recurrent_layer2 = tf.keras.layers.LSTM(params['hidden_size'], activation = 'tanh', return_sequences = True)
        else:
            self.recurrent_layer = tf.keras.layers.SimpleRNN(params['hidden_size'], activation = 'tanh', return_sequences = True)
        self.skill_layer = tf.keras.layers.Dense(params['num_skills'], activation = 'linear', name='skill_layer') 
        if params['1pl']: # decoder weights matrix is just Q-matrix
            print("\n 1PL \n")
            self.problem_layer = tf.keras.layers.Dense(params['num_items'], activation = 'sigmoid',
                                            kernel_constraint = q_1pl_constraint(self.q_matrix), name='problem_layer')
        else: 
            self.problem_layer = tf.keras.layers.Dense(params['num_items'], activation = 'sigmoid',
                                            kernel_constraint = q_constraint(self.q_matrix), name='problem_layer') 

    
    def get_config(self):
        return {'params': self.params}
    
    def get_skills(self, prob_x):
        x = self.embedding_layer(prob_x)
        x = self.recurrent_layer(x)
        x = self.recurrent_layer2(x)
        skills = self.skill_layer(x)
        return skills
    
    def call(self, inputs, training = False):
        prob_x, prob_y2 = inputs
        
        # need to get next problem ids (with no right/wrong)
        next_prob_id = tf.cast((prob_y2 - 4) // 2, dtype=tf.int32)
        
        # infer input size
        batch_size = tf.shape(prob_x)[0]
        sequence_length = tf.shape(prob_x)[1]
        
        with tf.name_scope('DKT_IRT'):
            embedded_x = self.embedding_layer(prob_x)
            recurrent = self.recurrent_layer(embedded_x) 
            if training:
                recurrent = tf.nn.dropout(recurrent, rate=self.params['dropout'])
            # Trying two LSTM layers
            recurrent = self.recurrent_layer2(recurrent)
            if training:
                recurrent = tf.nn.dropout(recurrent, rate=self.params['dropout'])
            skills = self.skill_layer(recurrent)
            problems = self.problem_layer(skills) # this should have shape (batch, seq_len, num_probs)
            mask = tf.one_hot(next_prob_id, depth=self.params['num_items'])
            output = tf.einsum('abi,abi->ab', problems, mask)
            return output # output shape (batch, seq_len)


def q_constraint(Q):
    def constraint(W):
        target = W * Q 
        diff = W - target
        W = W * tf.keras.backend.cast(tf.keras.backend.equal(diff, 0), tf.keras.backend.floatx())
        return W * tf.keras.backend.cast(tf.keras.backend.greater_equal(W, 0), tf.keras.backend.floatx()) 
    return constraint

def q_1pl_constraint(Q):
    def constraint(W):
        return Q
    return constraint
