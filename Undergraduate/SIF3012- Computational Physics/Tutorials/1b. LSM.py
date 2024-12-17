import numpy as np
import matplotlib.pyplot as plt

# Define the system of ODEs
def f1(x, y1, y2):
    return y2

def f2(x, y1, y2):
    return 4 * (y1 - x)

# Analytical solution
def y1_exact(x):
    return (-np.exp(4) * x + x + np.exp(2 - 2 * x) - np.exp(2 * x + 2)) / (1 - np.exp(4))

def y2_exact(x):
    return (-np.exp(4) + 1 - 2 * np.exp(2 - 2 * x) - 2 * np.exp(2 * x + 2)) / (1 - np.exp(4))

# Parameters for the Boundary Value Problem (BVP)
a, b = 0, 1
g_0, g_1 = 0, 2

# Define the general solver for different values of h
def LSM(h):
    n = int((b - a) / h)
    # Runge-Kutta 4th order method to solve the system
    def RK4(y1_0, y2_0):
        y1, y2 = y1_0, y2_0
        x = a
        
        x_values = [x]
        y1_values = [y1]
        y2_values = [y2]
        
        for i in range(n):
            k1_y1 = h * f1(x, y1, y2)
            k1_y2 = h * f2(x, y1, y2)

            k2_y1 = h * f1(x + h / 2, y1 + k1_y1 / 2, y2 + k1_y2 / 2)
            k2_y2 = h * f2(x + h / 2, y1 + k1_y1 / 2, y2 + k1_y2 / 2)

            k3_y1 = h * f1(x + h / 2, y1 + k2_y1 / 2, y2 + k2_y2 / 2)
            k3_y2 = h * f2(x + h / 2, y1 + k2_y1 / 2, y2 + k2_y2 / 2)

            k4_y1 = h * f1(x + h, y1 + k3_y1, y2 + k3_y2)
            k4_y2 = h * f2(x + h, y1 + k3_y1, y2 + k3_y2)

            y1 += (k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1) / 6
            y2 += (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2) / 6
            x += h

            x_values.append(x)
            y1_values.append(y1)
            y2_values.append(y2)
        
        return x_values, y1_values, y2_values

    # Newton-Raphson method to adjust the initial slope
    def newton_raphson(y1_0, y2_0, tol=1e-6, max_iter=10):
        for i in range(max_iter):
            # Solve the system with the current guess
            x_values, y1_values, y2_values = RK4(y1_0, y2_0)
            
            # Check the residual (difference at boundary x = b)
            residual = y1_values[-1] - g_1
            print(f"Iteration {i+1}, y2_0: {y2_0:.6f}, Residual: {residual:.6f}")
            
            if abs(residual) < tol:
                break
            
            # Numerical derivative of the residual with respect to y2_0
            delta = 1e-5
            _, y1_values_delta, _ = RK4(y1_0, y2_0 + delta)
            residual_prime = (y1_values_delta[-1] - y1_values[-1]) / delta

            # Update y2_0
            y2_0 -= residual / residual_prime
        
        return x_values, y1_values, y2_values

    # Initial guess for y2_0
    y2_0_guess = (g_1 - g_0) / (b - a)

    # Solve the BVP
    x_values, y1_values, y2_values = newton_raphson(g_0, y2_0_guess)

    # Plotting the results
    plt.figure(figsize=(16, 10))
    plt.scatter(x_values, y1_values, label="Approx. u", marker='o', color='red')
    plt.scatter(x_values, y2_values, label="Approx. u'", marker='x', color='blue')
    for i in range (0, n + 1):
        plt.text(x_values[i], y1_values[i], f'{y1_values[i]:.4f}', ha='right', va='bottom', color= 'red', fontsize=8)
    
    # Plot the exact solution
    x_exact = np.linspace(a, b, 1000)
    plt.plot(x_exact, y1_exact(x_exact), label="Exact u: $u(x) = \\frac{-e^4x+x+e^{2-2x}-e^{2x+2}}{1-e^4}$", linestyle='--', color='red')
    plt.plot(x_exact, y2_exact(x_exact), label="Exact u': $u'(x) = \\frac{-e^4+1-2e^{2-2x}-2e^{2x+2}}{1-e^4}$", linestyle='--', color='blue')
    
    plt.xlabel("x", fontsize = 16)
    plt.ylabel("y", fontsize = 16)
    plt.legend(fontsize = 12)
    plt.grid(True)
    plt.title(f"Solution for $u''=4(u - x)$ using Linear Shooting Method, for n={n}", fontsize = 20)
    plt.tight_layout()
    plt.show()

    # Print final values at each step
    for i in range(len(x_values)):
        print(f"Step {i}, x = {x_values[i]:.4f}, y = {y1_values[i]:.4f}, y' = {y2_values[i]:.4f}")

np.set_printoptions(edgeitems=10, linewidth=1000)

for h in [1/4, 1/8, 1/16]:
    LSM(h)
    print("---------------------------------------------------")
