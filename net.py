import numpy as np

class Net:
    @staticmethod
    def __to2dvec(arr):
        return np.ndarray((1, len(arr)), buffer=np.array(arr))
    
    def __init__(self, shape, eta):
        self.weights = [np.random.randn(i, o) for i, o in zip(shape[:-1], shape[1:])]
        self.biases = [np.zeros((1, l)) for l in shape[1:]]
        self.eta = eta
        
    def ffw(self, entry):
        entry = Net.__to2dvec(entry)
        outs = [entry]
        for w, b in zip(self.weights, self.biases):
            entry = np.tanh(np.dot(entry, w) + b)
            outs.append(entry)
        return entry, outs
    
    def bpp(self, outs, expect):
        expect = Net.__to2dvec(expect)
        
        error = outs[-1] - expect
        
        e_w = np.dot(outs[-2].T, error)
        e_b = np.sum(error, axis=0, keepdims=True)
        
        iter_len = len(outs)-1
        for i in reversed(range(1, iter_len)):
            error = np.dot(error, self.weights[i].T) * (1 - np.power(outs[i], 2))
            
            self.weights[i] -= self.eta * e_w
            self.biases[i] -= self.eta * e_b
            
            e_w = np.dot(outs[i-1].T, error)
            e_b = np.sum(error, axis=0, keepdims=True)
            
        self.weights[0] -= self.eta * e_w
        self.biases[0] -= self.eta * e_b
        
    def save(self, name):
        with open(name+".weights.pickle", "wb") as file: np.savez(file, *self.weights)
        with open(name+".biases.pickle", "wb") as file: np.savez(file, *self.biases)
        
    def load(self, name):
        with open(name+".weights.pickle", "rb") as file:
            self.weights = np.load(file, allow_pickle=True)
            self.weights = [self.weights[i] for i in self.weights]
            
        with open(name+".biases.pickle", "rb") as file:
            self.biases = np.load(file, allow_pickle=True)
            self.biases = [self.biases[i] for i in self.biases]
    
    def fit(self, entry, expect):
        self.bpp(self.ffw(entry)[1], expect)
        