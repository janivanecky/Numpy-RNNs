import numpy as np

def get_input(filename, length):
    f = open(filename,'r')
    input = f.read().decode('utf-8-sig').encode("ascii","ignore")
    if length < 0:
        length = len(input)
    input = input[0:length]
    input = list(input)
    chars = list(set(input))
    vocabulary_size = len(chars)
    char_to_index = { ch:i for i,ch in enumerate(chars) }
    index_to_char = { i:ch for i,ch in enumerate(chars) }

    return input, vocabulary_size, char_to_index, index_to_char

def char_to_one_hot(string, vocabulary_size, char_to_index):
    result = []
    for c in string:
        one_hot = np.zeros((vocabulary_size))
        one_hot[char_to_index[c]] = 1
        #one_hot[0] = 1
        result.append(one_hot)
    return np.array(result)

def batch_input_one_hot(string, start_index, length, batch_size, vocabulary_size, char_to_index):
    result = [char_to_one_hot(string[start_index + length * b: start_index + length * (b + 1)], vocabulary_size, char_to_index) for b in xrange(batch_size)]
    new_index = start_index + length * batch_size
    if(new_index + length * batch_size + 1 >= len(string)):
		new_index = 0
    return np.array(result), new_index

def batch_input(string, start_index, length, batch_size, char_to_index):
    result = [[char_to_index[c] for c in string[start_index + length * b: start_index + length * (b + 1)]] for b in xrange(batch_size)]
    new_index = start_index + length * batch_size
    if(new_index + length * batch_size + 1 >= len(string)):
		new_index = 0
    return np.array(result), new_index


def batch_input_one_hot_seq(string, start_index, length, batch_size, vocabulary_size, char_to_index):
    batch_part = len(string) / batch_size
    result = [char_to_one_hot(string[start_index + batch_part * b: start_index + length + batch_part * b], vocabulary_size, char_to_index) for b in xrange(batch_size)]
    new_index = start_index + length
    if(new_index + length + 1 >= batch_part):
		new_index = 0
    return np.array(result), new_index
    
def batch_input_seq(string, start_index, length, batch_size, char_to_index):
    batch_part = len(string) / batch_size
    result = [[char_to_index[c] for c in string[start_index + batch_part * b: start_index + batch_part * b + length]] for b in xrange(batch_size)]
    new_index = start_index + length
    if(new_index + length + 1 >= batch_part):
		new_index = 0
    return np.array(result), new_index
