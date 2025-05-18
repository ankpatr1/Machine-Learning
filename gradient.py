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

# Plot results - COMBINED PLOT
plt.figure(figsize=(8, 6))  # Adjust figure size if needed
plt.plot(f_values1, label='η = 0.01')
plt.plot(f_values2, label='η = 0.1')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Gradient Descent Progress')
plt.grid(True)  # Add grid lines
plt.legend()
plt.show()

# # --- Part B: Find minimum values from different starting points (COMMENTED OUT) ---
# starting_points = [(0.1, 0.1), (1, 1), (0.5, 0.5), (0.0, 0.5), (-0.5, -0.5), (-1, 1)]
#
# print("\n--- Part B: Minimum Values from Different Starting Points ---")
# for i, (x0, y0) in enumerate(starting_points):
#     x_min, y_min, _ = gradient_descent(x0, y0, 0.01, 50)
#     min_value = f(x_min, y_min)
#     print(f"Starting Point {i+1}: ({x0}, {y0})")
#     print(f"Minimum Value: {min_value:.4f}")
#     print(f"Location: ({x_min:.4f}, {y_min:.4f})")

print("\nObservations:")
print("  - Smaller learning rate (η=0.01) leads to a smoother, more stable descent, but may converge slower.")
print("  - Larger learning rate (η=0.1) can lead to faster initial descent, but risks overshooting the minimum, resulting in oscillations.")
