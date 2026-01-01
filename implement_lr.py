import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(x)

for i in range(epochs):
    
    y_pred = m * x + b
    
    loss = (1/n) * np.sum((y_pred - y) ** 2)
    
    dm = (2/n) * np.sum((y_pred - y) * x)
    db = (2/n) * np.sum(y_pred - y)
    
    m = m - learning_rate * dm
    b = b - learning_rate *db
    
    if i % 100 == 0:
        print(f"epoch {i} : loss = {loss:.4f}, m={m:.4f}, b={b:.4f}")
    
    
    