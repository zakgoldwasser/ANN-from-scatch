import numpy as np
import time


n_hidden = 10
n_hidden2 = 10
n_in = 10

n_out = 10

n_sample = 300

learning_rate = 0.01
momentum = 0.9

np.random.seed(0)

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
	return 1- np.tanh(x)**2

def train(x, t, V, V2, W, bv, bv2, bw): 
	'''x=input data
	t=transpose
	V=layer1
	V2=layer2
	W=layer3
	bv=bias for v
	bv=2bias for v2
	bw= bias for W
	'''
	#forward - matrix mult+bias
	A = np.dot(x, V) + bv #layer1 
	Z = np.tanh(A) #relu

	A2 = np.dot(Z, V2) + bv2 #layer2 
	Z2 = np.tanh(A2) #relu

	B= np.dot(Z2, W) + bw #layer3
	Y= sigmoid(B) #relu
	#print("y",Y)

	#backward prop
	#print("t",t)
	Ew = Y - t #output - flipped matrix
	#print(Ew)
	Ev = tanh_prime(A2) * np.dot(W, Ew) #(matrix multiply layer 3 by Ew) * tanh_prime
	Ev2 = tanh_prime(A) * np.dot(V2, Ev) #(matrix multiply layer 2 by Ev) * tanh_prime 
	#print("EV",Ev)

	#predict loss
	dW = np.outer(Z2, Ew) #(l2 activation output * Ew)
	dV = np.outer(Z, Ev) #mult Z[0] * Ew[0] (l1 activation output * Ew)
	dV2 = np.outer(x, Ev2) #mult x[0] * EV[0] (input_data * )
	

	#cross entropy (for classification)
	loss = -np.mean(t * np.log(Y)+(1-t)*np.log(1-Y))

	return loss, (dV2, dV, dW, Ev2, Ev, Ew)

def predict(x, V, V2, W, bv, bv2, bw):
	A = np.dot(x,V) + bv
	B = np.dot(np.tanh(A),V2) + bv2
	C = np.dot(np.tanh(B), W) + bw
	return(sigmoid(C) > 0.5).astype(int)

#create layers
V = np.random.normal(scale=0.1, size=(n_in, n_hidden)) #random numbers (in X layer)
V2 = np.random.normal(scale=0.1, size=(n_hidden, n_hidden2)) #random numbers (in X layer)
W = np.random.normal(scale=0.1, size=(n_hidden2, n_out)) #random numbers (layer X out)


bv = np.zeros(n_hidden) # list of zeros len jidden nodes
bv2 = np.zeros(n_hidden2)
bw = np.zeros(n_out) # list of zeros len output



params= [V, V2, W, bv, bv2, bw] #list of params


#generates data
X= np.random.binomial(1, 0.5, (n_sample, n_in)) #data
T = X ^ 1 #label


#Training
for epoch in range(100):
	err = []
	update = [0]*len(params)

	t0=time.clock()
	for i in range(X.shape[0]):
		loss,grad = train(X[i], T[i], *params)
		for j in range(len(params)):
			params[j] -= update[j] #updates params(layers and weoghts)
		for j in range(len(params)):
			update[j] = learning_rate * grad[j] + momentum * update[j] #value at grad point affected by hyper params

		err.append(loss)

	print('Epoch: %d, Loss: %.8f, Time: %.4fs'%(
		epoch, np.mean(err),time.clock()-t0))
x = np.random.binomial(1, 0.5, n_in)
print('XDR prediction')
print(x)
print(predict(x, *params))