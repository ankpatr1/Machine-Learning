import numpy as np
import matplotlib.pyplot as plt

# Define the function to minimize
def f(x, y):
    return 2*x**2 + y**2 + 3*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

# Define the gradient of the function
def grad_f(x, y):
    dx = 4*x + 6*np.pi*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
    dy = 2*y - 6*np.pi*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    return np.array([dx, dy])

# Implement the gradient descent algorithm
def gradient_descent(x0, y0, learning_rate, num_iterations):
    x, y = x0, y0
    f_values = []
    x_values = []  # Store x values for plotting (optional)
    y_values = []  # Store y values for plotting (optional)
    for _ in range(num_iterations):
        f_values.append(f(x, y))
        x_values.append(x)  # store x value
        y_values.append(y)  # store y value
        grad = grad_f(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
    return x_values, y_values, f_values #Return x and y values for plotting


# Set parameters
start_x = 0.1
start_y = 0.1
learning_rate_1 = 0.01
learning_rate_2 = 0.1
num_iterations = 50

# Run gradient descent for η = 0.01
x_values_1, y_values_1, f_values_1 = gradient_descent(start_x, start_y, learning_rate_1, num_iterations)

# Run gradient descent for η = 0.1
x_values_2, y_values_2, f_values_2 = gradient_descent(start_x, start_y, learning_rate_2, num_iterations)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot for η = 0.01
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.plot(f_values_1)
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("Gradient Descent (η = 0.01)")

# Plot for η = 0.1
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.plot(f_values_2)
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("Gradient Descent (η = 0.1)")

plt.tight_layout()  # Adjust subplot parameters for a tight layout
plt.show()

# --- Part B: Find minimum values from different starting points ---
starting_points = [(0.1, 0.1), (1, 1), (0.5, 0.5), (0.0, 0.5), (-0.5, -0.5), (-1, 1)]

print("\n--- Part B: Minimum Values from Different Starting Points ---")
for i, (x0, y0) in enumerate(starting_points):
    x_values, y_values, f_values = gradient_descent(x0, y0, learning_rate_1, num_iterations)
    x_min = x_values[-1]  # Last x value
    y_min = y_values[-1]  # Last y value
    min_value = f(x_min, y_min)
    print(f"Starting Point {i+1}: ({x0}, {y0})")
    print(f"Minimum Value: {min_value:.4f}")  # Format to 4 decimal places
    print(f"Location: ({x_min:.4f}, {y_min:.4f})")  # Format to 4 decimal places

print("\nObservations:")
print("  - Smaller learning rate (η=0.01) leads to a smoother, more stable descent, but may converge slower.")
print("  - Larger learning rate (η=0.1) can lead to faster initial descent, but risks overshooting the minimum, resulting in oscillations.")
print("  - The starting point significantly affects the final minimum found, demonstrating that this function has multiple local minima.")
