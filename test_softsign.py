import numpy as np

x = np.linspace(-1, 1, num=100)
y = x * 16 / (1 + np.abs(x) * 16)

for a, b in zip(x, y):
    print(f"{b}")