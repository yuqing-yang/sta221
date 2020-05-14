
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

def power_iteration(A,v0,niter=50):
    v = v0
    vv = [v0]
    ll = [np.dot(v0,np.dot(A,v0))]
    for k in range(niter):
        w = np.dot(A,v)
        v = w/np.linalg.norm(w)
        lamda = np.dot(v,np.dot(A,v))
        vv.append(v)
        ll.append(lamda)
    return ll, vv

A = np.random.rand(5,5)*5 # generate a 5-by-5 matrix A
start = time.time()
ll, vv = power_iteration(A,np.ones(5))
end = time.time()
print('timecost:',end - start)
plt.figure(figsize=(3, 3))
plt.plot(range(len(ll)),ll,'-o')
plt.ylabel('Eigenvalue approximation')
plt.xlabel('Iteration');
plt.savefig('smallMatrix')
plt.close()

m = 1000
A = np.random.rand(m,m)*5 # generate a 5-by-5 matrix A
start = time.time()
ll, vv = power_iteration(A,np.ones(m))
end = time.time()
print('timecost:',end - start)
plt.figure(figsize=(3, 3))
plt.plot(range(len(ll)),ll,'-o')
plt.ylabel('Eigenvalue approximation')
plt.xlabel('Iteration');
plt.savefig('largeMatrix')
plt.close()

np.save('lambda',ll)

