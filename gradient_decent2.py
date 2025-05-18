import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return 2*x**2 + y**2 + 3*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

def grad_f(x, y):
    dx = 4*x + 6*np.pi*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
    dy = 2*y - 6*np.pi*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    return np.array([dx, dy])

def gradient_descent(x0, y0, learning_rate, num_iterations):
    x, y = x0, y0
    f_values = []
    for _ in range(num_iterations):
        f_values.append(f(x, y))
        grad = grad_f(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
    return x, y, f_values

# Run gradient descent with η = 0.01
x_min1, y_min1, f_values1 = gradient_descent(0.1, 0.1, 0.01, 50)

# Run gradient descent with η = 0.1
x_min2, y_min2, f_values2 = gradient_descent(0.1, 0.1, 0.1, 50)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(f_values1)
plt.title('η = 0.01')
plt.xlabel('Iteration')
plt.ylabel('Function Value')

plt.subplot(122)
plt.plot(f_values2)
plt.title('η = 0.1')
plt.xlabel('Iteration')
plt.ylabel('Function Value')

plt.tight_layout()
plt.show()
starting_points = [(0.1, 0.1), (1, 1), (0.5, 0.5), (0.0, 0.5), (-0.5, -0.5), (-1, 1)]

for i, (x0, y0) in enumerate(starting_points, 1):
    x_min, y_min, _ = gradient_descent(x0, y0, 0.01, 50)
    min_value = f(x_min, y_min)
    print(f"Starting point {i}: ({x0}, {y0})")
    print(f"Minimum value: {min_value}")
    print(f"Location: ({x_min}, {y_min})")
    print()
