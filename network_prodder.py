import math
import pickle

import numpy as np

LOGDIR = '005'
ALT_LOGDIR = '008'

IN1 = 'two'
IN2 = 'four'

DIGS = sorted(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
               'eight', 'nine', 'ten', 'over', 'under'])

DIG_TO_IX = {dig: DIGS.index(dig) for dig in DIGS}
IX_TO_DIG = {ix: DIGS[ix] for ix in range(len(DIGS))}

def one_hot_encoding_of(dig):
    out = [0 for i in DIGS]
    out[DIG_TO_IX[dig]] = 1
    return np.array(out)

def unpickled_network(logdir):
    with open('logdirs/logdir_{}/runlog_0.pkl'.format(logdir),
              'rb') as pkl:
        out = pickle.load(pkl)
    return out['test_data'][-1]

def weights_from_network(unpickled_net, layer_names):
    out = {}
    for layer in layer_names:
        out[layer] = unpickled_net[layer]['weights']
    return out

def biases_from_network(unpickled_net, layer_names):
    out = {}
    for layer in layer_names:
        out[layer] = unpickled_net[layer]['biases']
    return out
        
def calculate_activations(logdir):
    layers = ['a1h', 'a1o', 'a1r', 's1h', 's1o', 's2h', 's2o', 'out']

    unpickled_net = unpickled_network(logdir)
    weights = weights_from_network(unpickled_net, layers)
    biases = biases_from_network(unpickled_net, layers)

    deps = {
        'a1h': ['in1', 'in2'],
        'a1o': ['a1h'],
        'a1r': ['a1o'],
        's1h': ['a1r', 'in1'],
        's1o': ['s1h'],
        's2h': ['in2', 's1o'],
        's2o': ['s2h'],
        'out': ['s2o', 'a1o']
    }

    activations = {layer: None for layer in layers}
    activations['in1'] = one_hot_encoding_of(IN1)
    activations['in2'] = one_hot_encoding_of(IN2)

    # If you want to modify any weight or bias manually, do it here!


    for layer in layers:
        incoming = np.array([])
        for segment in deps[layer]:
            incoming = np.concatenate((incoming, activations[segment]))
        net = np.dot(weights[layer], incoming)
        net = net + biases[layer]
        sigmoid = np.vectorize(lambda x: 1/(1+math.exp(-x)))
        net = sigmoid(net)

        activations[layer] = net
        #print('Layer {} activations:'.format(layer))
        #print(net)
        #print('\n\n')
    return activations

def main():
    act1 = calculate_activations(LOGDIR)
    act2 = calculate_activations(ALT_LOGDIR)
    print(act1['s2o'])
    print(act2['s2o'])

if __name__ == '__main__':
    main()
