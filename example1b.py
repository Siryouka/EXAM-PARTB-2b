import numpy as np
import matplotlib.pyplot as plt

np.random.seed()

a = 1
b = 1
m0 = 1
m1 = 20
m2 = 10
m3 = 1
eta1 = 0.1

w1 = 0.2 * np.random.rand(m1, m0 + 1)
w2 = 0.2 * np.random.rand(m2, m1 + 1)
w3 = 0.2 * np.random.rand(m3, m2 + 1)

yp = np.array([0, 0])
ym = np.array([0, 0])

for k in range(1, 50000):
    if k <= 10000:
        u = -1 + 2 * np.random.rand(1)  #### random value u is generated between -1 and 1



        a = np.array([1], dtype = np.float32)
        p = np.concatenate([a, u], axis=0).reshape(-1, 1)
        #p = np.array([1, u]).reshape(-1, 1)

        v1 = np.matmul(w1, p) ####p is fed through three layers of the neural network (w1, w2, and w3)
        v11 = b * v1
        # phi_v1 = a * np.tanh(b * v1)

        phi_v1 = a * np.tanh(v11)   ##nonlinear activation functions applied at each layer
        #a = np.array([1], dtype=np.float32)
        y1_k = np.concatenate((a.reshape(-1, 1), phi_v1))
        v2 = np.dot(w2, y1_k)
        phi_v2 = a * np.tanh(b * v2)
        y2_k = np.concatenate((a.reshape(-1, 1), phi_v2))
        v3 = np.dot(w3, y2_k)
        y3 = v3
        E = u ** 3 + 0.3 * u ** 2 - 0.4 * u - y3  ##defines a target function E based on the input value u and the output of the neural network y3
        phi_v3_diff = 1
        phi_v2_diff = a * (1 - phi_v2 ** 2)
        phi_v1_diff = a * (1 - phi_v1 ** 2)

        # BACKWARD PASS
        delta3 = E * phi_v3_diff  ##performs the backward pass to update the weights of the neural network in order to minimize the error
        delta_w3 = eta1 * np.outer(delta3, y2_k)
        delta2 = np.matmul(w3[0, 1:].reshape(1, -1).T, delta3) * phi_v2_diff
        delta_w2 = eta1 * np.outer(delta2, y1_k)
        delta1 = np.dot(w2[:, 1:].T, delta2) * phi_v1_diff
        delta_w1 = eta1 * np.outer(delta1, p)

        # ERROR CALCULATION AND WEIGHT UPDATION
        w1 = w1 + delta_w1
        w2 = w2 + delta_w2
        w3 = w3 + delta_w3

# Testing the trained network
for k in range(1, 1002):
    u = np.sin(2 * np.pi * k / 250) + np.sin(2 * np.pi * k / 25)
    p = np.array([1, u]).reshape(-1, 1)

    y1 = 0.3 * yp[-1] + 0.6 * yp[-2] + u ** 3 + 0.3 * u ** 2 - 0.4 * u

    # FORWARD PASS
    v1 = np.dot(w1, p)
    phi_v1 = a * np.tanh(b * v1)
    a = np.array([1], dtype = np.float32)
    y1_k = np.concatenate((a.reshape(1,1), phi_v1))
    v2 = np.matmul(w2, y1_k)
    phi_v2 = a * np.tanh(b * v2)
    y2_k = np.concatenate((a.reshape(1,1), phi_v2))
    v3 = np.matmul(w3, y2_k)
    y2 = v3 + 0.3 * yp[-1] + 0.6 * yp[-2]

    yp = np.append(yp, y1)
    ym = np.append(ym, y2)

K = np.arange(1003)
plt.plot(K, ym, '-r', label='Neural Network Output')
plt.plot(K, yp, '--g', label='Desired Output')
plt.legend()
plt.title('Example 1b Function Approximation')
plt.show()

