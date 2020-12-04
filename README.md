# CS-GY 6923 Final Project

## Extention 1 SVM with Soft margein

In real world, some data is not linearly separable. And at this situation, soft margin can increase the distance between the margins and therefore can fit more data, tolerate more noise and make the model more robust. 

The formula is :

`y (i)(w Tx (i)+ w 0) ≥1− ξ(i)  ξ>.0`

We use `ξ` to describe how far x(i) away from the margin. We use varibale `C` to represent `ξ` in the code

### Scikit-learn implementation:

```python
model = svm.SVC(kernel='linear', C=C[i], gamma='auto')
model.fit(X[train], y[train])
y_pred = model.predict(X_test)
res.append(accuracy_score(y_test, y_pred) * 100)
```

### Numpy implementation:

```python
def kernel_svm_soft(X, y, C):
    N = len(y)
    alphas = []
    y = y.reshape(-1, 1)
    yX = y * X
    
    P = matrix(yX.dot(yX.T))
    q = matrix((-1) * np.ones(X.shape[0]).reshape((X.shape[0], 1)))

    G = matrix(np.vstack((-1 * np.diag(np.ones(X.shape[0])), np.identity(X.shape[0]))))
    h = matrix(np.hstack((np.zeros(X.shape[0]), np.ones(X.shape[0])*C)))

    A = matrix(y.reshape(1,-1))
    A = matrix(A, (1, X.shape[0]), 'd')
    b = matrix(np.zeros(1))
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x']).flatten()
    return alphas
alphas = kernel_svm_soft(X_train, y_train, 0.001)

```

#### Extension 1 Statistic

 Implementaion | Dataset | Accuracy 
--- | --- | ---
homework implementation | homework CSV | 100%
Soft margin implementation | homework CSV | 100%
homework implementation |scikit-learn digits | 87.5%
Soft margin implementation |scikit-learn digits | 90.28%


## Extension 2 Nerual Network with Softmax Output Layer

People usually use `sigmoid` activation on multi-lable classification problem, the outpus are not mutually exclusive. The `sigmoid` function gives the probablity for all of the classes. The results can have high probability for all of the classes or none of them. However, when dealing with mutually exclusive outputs, or need to find out the only right answers type of questions, `softmax` function can be a better choice. The `softmax` function enforces the sum of the probabilities of output classes equal to one, and increased the probability of a particular class. In this homework, we are asked to find out the only one correct digit from 10 different digits, therefore we need to increase one particular class's probability, and `softmax` function can be a good choice here.

## Tensorflow Implementation
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation="sigmoid"))
model.add(Dense(30, activation = "sigmoid"))
model.add(Dense(10, activation = "sigmoid"))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics = ['accuracy'])
model.fit(X_train, y_v_train, epochs = 120, batch_size = 1)

accuracy = model.evaluate(X_test, y_v_test)
print('Accuracy: %.2f' % (accuracy[1]*100))
```

##Numpy Implementation

```python

def softmax(z):
    return np.exp(z - np.max(z))/np.sum(np.exp(z - np.max(z)), axis=0, keepdims=True) 

def feed_forward_softmax(x, W, b, nn_structure):
    a = {1: x} # create a dictionary for holding the a values for all levels
    z = { } # create a dictionary for holding the z values for all the layers
    for l in range(1, len(W) + 1): # for each layer
        node_in = a[l]
        z[l+1] = W[l].dot(node_in) + b[l]  # z^(l+1) = W^(l)*a^(l) + b^(l)
        #check if last layer is softmax layer
        if z[l+1].shape[0] == nn_structure[-1]:
            a[l+1] = softmax(z[l+1])
        else:
            a[l+1] = f(z[l+1]) # a^(l+1) = f(z^(l+1))
    return a, z

def calculate_out_layer_delta_softmax(y, a_out, z_out):
    return -(y-a_out)

# detailed code please check the Extention2 folder
```
#### Extension 2 Statistic

Implementaion | Dataset | Accuracy 
--- | --- | ---
homework implementation | scikit-learn digits | 89.98%
Softmax implementation | scikit-learn digits | 95.96%
homework implementation |scikit-learn wines | 61.12%
Softmax implementation |scikit-learn wines | 97.22%

