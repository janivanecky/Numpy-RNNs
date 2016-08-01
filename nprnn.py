'''
Simple NPRNN model in python/numpy, written by Jan Ivanecky (@janivanecyk) 

http://arxiv.org/pdf/1511.03771v3.pdf

MIT license
'''

import numpy as np
import math
import util
from random import uniform
from graph import Grapher

# softmax layer
def softmax(bottom):
	top = np.exp(bottom) / np.sum(np.exp(bottom))
	return top

# note: this is not a real cross entropy loss implementation, it's a simplified version
# built with assumption that the ground truth vector contains only one non-zero component with a value of 1
# gt_index is the index of that non-zero component
def cross_entropy(bottom, gt_index):
	loss = -np.log(bottom[gt_index]) if bottom[gt_index] > 0 else -np.log(1e-8)
	return loss

# note: once again, this function does not compute a general derivative of a softmax followed by a cross entropy loss,
# it computes a derivative for this special case
def cross_entropy_softmax_d(top, gt_index):
	d = np.copy(top)
	d[gt_index] -= 1
	return d

# relu activation
def relu(bottom):
	top = bottom * (bottom > 0)
	return top

# computes derivative of a relu activation with respect to its inputs
def relu_d(bottom):
	d = bottom > 0
	return d

# tanh activation
def tanh(bottom):
	top = np.tanh(bottom)
	return top

# computes derivative of a tanh activation with respect to its inputs 
def tanh_d(top):
	d = 1 - top * top 

# initialize input matrix with a Xavier method
def input_matrix_init(input_size, hidden_size):
	stdev = np.sqrt(2.0 / (input_size + hidden_size))
	Wxh = np.random.randn(input_size, hidden_size) * stdev
	return Wxh

# intialize recurrent weights by positive definitive matrix with all except the highest eigenvalue < 0 (Talathi et al. http://arxiv.org/pdf/1511.03771v3.pdf)
def recurrent_matrix_init_NPRNN(hidden_size):
	R = np.random.randn(hidden_size,hidden_size) 
	A = np.dot(R.T, R) / float(hidden_size)
	I = np.identity(hidden_size)
	e = (np.linalg.eigvals(A + I)).max()
	Whh = (A + I) / e
	return Whh

# initialize recurrent weights with an identity matrix (Le et al. http://arxiv.org/pdf/1504.00941v2.pdf) 
def recurrent_matrix_init_IRNN(hidden_size):
	Whh = np.identity(hidden_size)
	return Whh

# initialize recurrent weights with a Xavier method
def recurrent_matrix_init_basic(hidden_size):
	stdev = np.sqrt(2.0 / (hidden_size + hidden_size))
	Whh = np.random.randn(hidden_size, hidden_size) * stdev
	return Whh

# hyperparameters
HIDDEN_LAYER_SIZE = 256
DEPTH = 3
DROPOUT_RATE = 0.1

SEQ_SIZE = 100
BATCH_SIZE = 1
L_RATE = 0.01
MAX_ITERATIONS = 100000
EVAL_INTERVAL = 100
PRINT_SAMPLES = True
TEMPERATURE = 0.7

# get input
input, VOCABULARY_SIZE, char_to_index, index_to_char = util.get_input('shakespear_train.txt', -1)
validation_input, _, _, _, = util.get_input('shakespear_val.txt', 5000)
# model parameters
Wxh = [input_matrix_init(VOCABULARY_SIZE if d == 0 else HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) for d in xrange(DEPTH)]
Whh = [recurrent_matrix_init_NPRNN(HIDDEN_LAYER_SIZE) for d in xrange(DEPTH)]
bh = [np.zeros((1, HIDDEN_LAYER_SIZE)) for d in xrange(DEPTH)]
Why = np.random.randn(HIDDEN_LAYER_SIZE, VOCABULARY_SIZE) * np.sqrt(2.0 / (VOCABULARY_SIZE + HIDDEN_LAYER_SIZE))
by = np.zeros((1, VOCABULARY_SIZE))

def forward_backward(inputs, targets, initial_states):
	'''
	Computes forward and backward pass through the recurrent net, for SEQ_SIZE time steps
	-inputs is an array of shape [BATCH_SIZE, SEQ_SIZE, VOCABULARY_SIZE] and holds one hot encoded inputs to the model
	-targets has a shape [BATCH_SIZE, SEQ_SIZE], holds just the indices of the target chars
	-initial_states contains state of all the recurrent hidden units, shape [DEPTH, BATCH_SIZE, HIDDEN_LAYER_SIZE]

	Returns loss, gradients and the last state of the hidden units
	'''
	loss = 0

	dropout = [{} for i in xrange(DEPTH)]
	x,h,z = [{} for i in xrange(DEPTH + 1)], [{} for i in xrange(DEPTH)], {}
	
	# Initialize states
	h = [{-1: initial_states[d]} for d in xrange(DEPTH)]

	# Forward pass
	for t in xrange(SEQ_SIZE):
		x[0][t] = np.reshape(inputs[:,t,:], (BATCH_SIZE, VOCABULARY_SIZE))
			
		for d in xrange(DEPTH):
			dropout[d][t] = np.random.binomial(1, 1 - DROPOUT_RATE, (1, HIDDEN_LAYER_SIZE)) * 1.0 / (1 - DROPOUT_RATE)
			h[d][t] = relu(np.dot(x[d][t], Wxh[d]) + np.dot(h[d][t - 1], Whh[d]) + bh[d])
			x[d + 1][t] = np.copy(h[d][t]) * dropout[d][t]
		y = np.dot(x[DEPTH][t], Why) + by
		y = np.clip(y, -100,100) # clipping to prevent state explosions at the beggining of training
		z[t] = np.array([softmax(y[b,:]) for b in xrange(BATCH_SIZE)])

	# Backward pass
	dWhy = np.zeros((HIDDEN_LAYER_SIZE, VOCABULARY_SIZE))
	dby = np.zeros((1, VOCABULARY_SIZE))
	dWxh = [np.zeros((VOCABULARY_SIZE if d == 0 else HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)) for d in xrange(DEPTH)]
	dWhh = [np.zeros((HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)) for d in xrange(DEPTH)]
	dbh = [np.zeros((1, HIDDEN_LAYER_SIZE)) for d in xrange(DEPTH)]
	dhh = [np.zeros((BATCH_SIZE, HIDDEN_LAYER_SIZE)) for i in xrange(DEPTH)]
	
	for t in reversed(xrange(SEQ_SIZE)):
		gt = targets[:, t]

		loss += np.array([cross_entropy(z[t][b, :], gt[b]) for b in xrange(BATCH_SIZE)]).sum() / (SEQ_SIZE * BATCH_SIZE)
		dy = np.array([cross_entropy_softmax_d(z[t][b, :], gt[b]) for b in xrange(BATCH_SIZE)]) / (SEQ_SIZE * BATCH_SIZE)
		
		dWhy += np.dot(x[DEPTH][t].T, dy)
		dby += dy.sum(0)
		dh = np.dot(dy, Why.T) 
		for d in reversed(xrange(DEPTH)):
			dhinput = relu_d(np.dot(x[d][t], Wxh[d]) + np.dot(h[d][t-1], Whh[d]) + bh[d]) * (dh * dropout[d][t] + dhh[d])
			dWxh[d] += np.dot(x[d][t].T, dhinput)
			dWhh[d] += np.dot(h[d][t-1].T, dhinput)
			dbh[d] += dhinput.sum(0)
			dhh[d] = np.dot(dhinput, Whh[d].T)
			dh = np.dot(dhinput, Wxh[d].T)
	
	h_prev = np.array([h[d][SEQ_SIZE - 1] for d in xrange(DEPTH)]) # get last states for the next train step
	return loss, dWxh, dWhh, dbh, dWhy, dby, h_prev

def forward(input, state):
	'''
	Computes only the forward pass through one step of the time, note that the input to the softmax is divided by a hyperparameter TEMPERATURE
	-input is an index of the char in vocabulary
	-state, the same as for forward_backward, but the BATCH_SIZE is 1, so the final shape is [DEPTH, 1, HIDDEN_LAYER_SIZE] 
	
	Returns probabilities and the updated state of the hidden units
	'''
	ox = np.zeros((1, VOCABULARY_SIZE))
	ox[0, input] = 1

	for d in xrange(DEPTH):
		state[d] = relu(np.dot(ox, Wxh[d]) + np.dot(state[d], Whh[d]) + bh[d])
		ox = state[d]
		
	y = np.dot(ox, Why) + by
	y = np.clip(y, -100, 100)
	oz = softmax(y / TEMPERATURE)
	return np.reshape(oz, (VOCABULARY_SIZE)), state

def evaluate_loss(input):
	'''
	Evaluates and returns loss on the input string (array of chars)
	'''
	oh = [np.zeros((1, HIDDEN_LAYER_SIZE)) for i in xrange(DEPTH)]
	loss = 0
	N = len(input) - 1
	for i in xrange(N):
		inpt = char_to_index[input[i]]
		target = char_to_index[input[i + 1]]
		prob, oh = forward(inpt, oh)
		target_prob = -np.log(prob[target]) / N
		loss += target_prob
	return loss
		
def sample_model(N):
	'''
	Samples the model, returns the sample of length N as a string
	'''
	ix = np.random.randint(0, VOCABULARY_SIZE)
	output = []
	output.append(index_to_char[ix])
	oh = [np.zeros((1, HIDDEN_LAYER_SIZE)) for i in xrange(DEPTH)]
	for c in xrange(N):
		oz, oh = forward(ix, oh)	
		result = np.random.choice(range(VOCABULARY_SIZE), p=oz.ravel())
		output.append(index_to_char[result])
		ix = result
	return ''.join(output).rstrip()
	
# initial states 
h_prev = np.array([np.zeros((BATCH_SIZE, HIDDEN_LAYER_SIZE)) for d in xrange(DEPTH)])
p = 0

# momentum for Adagrad
mWxh = [np.zeros((VOCABULARY_SIZE if d == 0 else HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)) + 0.1 for d in xrange(DEPTH)]
mWhh = [np.zeros((HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)) + 0.1 for d in xrange(DEPTH)]
mbh = [np.zeros((1, HIDDEN_LAYER_SIZE)) + 0.1 for d in xrange(DEPTH)]
mWhy = np.zeros((HIDDEN_LAYER_SIZE, VOCABULARY_SIZE)) + 0.1
mby = np.zeros((1, VOCABULARY_SIZE)) + 0.1

losses = {}
graph = Grapher('Train Loss')
# training loop
for iteration in xrange(MAX_ITERATIONS):
	# get inputs for current iteration
	if p == 0:
		h_prev = np.array([np.zeros((BATCH_SIZE, HIDDEN_LAYER_SIZE)) for d in xrange(DEPTH)])
	targets, _ = util.batch_input_seq(input, p + 1, SEQ_SIZE, BATCH_SIZE, char_to_index)
	t, _ = util.batch_input_seq(input, p, SEQ_SIZE, BATCH_SIZE, char_to_index)
	inputs, p = util.batch_input_one_hot_seq(input, p, SEQ_SIZE, BATCH_SIZE, VOCABULARY_SIZE, char_to_index)
	

	loss, dWxh, dWhh, dbh, dWhy, dby, h_prev = forward_backward(inputs, targets, h_prev)

	# update model parameters
	all_d = dWxh + dWhh + dbh + [dWhy, dby]
	all_m = mWxh + mWhh + mbh + [mWhy, mby]
	all_w = Wxh + Whh + bh + [Why, by]
	for d, m, w in zip(all_d, all_m, all_w):
		np.clip(d, -1, 1, out=d)
		# Adagrad
		m += d * d
		w -= L_RATE * d / np.sqrt(m + 1e-8)
		# RMSProp
		#m = 0.9 * m + 0.1 * d * d
		#w -= L_RATE * d / np.sqrt(m + 1e-8)
		
	# sample from the model and evaluate test loss
	if(iteration % EVAL_INTERVAL == 0):
		print("ITERATION " + str(iteration))
		print 'loss: {}'.format(loss * 25)
		
		# evaluate test loss
		validation_loss = evaluate_loss(validation_input)
		print('validation loss: {}'.format(validation_loss * 25))
		
		# sample the model
		if(PRINT_SAMPLES):
			output = sample_model(200)
			print(output)
		
		losses[iteration] = loss
		graph_keys = np.array(sorted(losses), dtype=np.uint32)
		graph_data = np.array([losses[key] for key in graph_keys], dtype=np.float32)
		graph.update(graph_keys, graph_data)


weights = [Wxh, Whh, bh, Why, by]
np.save('nprnn_weights.npy', weights)
