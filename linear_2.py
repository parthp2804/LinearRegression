from sklearn.datasets.samples_generator import make_regression
import numpy as np
X,y = make_regression(n_samples=200, n_features=1, n_informative=1, noise=6, bias=30, random_state=200)
m=200

def h(X,w):
	return (w[1]*np.array(X[:,0])+w[0])

def cost(w,X,y):
	return (.5/m) * np.sum(np.square(h(X,w) - np.array(y)))

def grad(w,X,y):
	g = [0]*2
	g[0] = (1/m)*np.sum(h(X,w) - np.array(y))
	g[1] = (1/m)*np.sum((h(X,w) - np.array(y))* np.array(X[:,0]))
	return g
def descent(w_new,w_prev,lr):
	print(w_prev)
	print(cost(w_prev,X,y))
	j=0
	while True:
		w_prev = w_new
		w0 = w_prev[0] - lr*grad(w_prev,X,y)[0]
		w1 = w_prev[1] - lr*grad(w_prev,X,y)[1]
		w_new = [w0,w1]
		print(w_new)
		print(cost(w_new,X,y))
		if ((w_new[0]-w_prev[0])**2 + (w_new[1]-w_prev[1])**2 )<= pow(10,-6):
			return w_new
		if j>500:
			return w_new
		j+=1

w = [0,-1]

w = descent(w,w,.1)
print(w)