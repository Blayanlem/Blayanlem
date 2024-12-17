# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# [CHANGE title and input] Define the source term function for the equation u'' = 4(u - x), rearranging into -u'' + 4u = 4x
def f(x):
    return 4 * x
# [CHANGE] Define the exact solution for comparison
def y(t):
    return (-np.exp(4) * t + t + np.exp(2 - 2 * t)- np.exp(2 * t + 2)) / (1 - np.exp(4))

# Parameters for the Boundary Value Problem:
a, b = 0, 1
g_0, g_1 = 0, 2
p, q = 0, 4

def bvp_soln(h):
    # Discretize the interval
    N = int((b - a) / h)
    x = np.linspace(a, b, N + 1)

    # Construct the tridiagonal matrix A for the system
    A = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        if i > 0:
            A[i, i - 1] = - (1 + h / 2 * p)
        A[i, i] = 2 + q * h ** 2
        if i < N - 2:
            A[i, i + 1] = - (1 - h / 2 * p)

    # Construct vector B for the system
    B = np.array([h ** 2 * f(x[i + 1]) for i in range(N - 1)])

    # Incorporate boundary conditions
    B[0] += (1 + 0.5 * p * h) * g_0
    B[-1] += (1 - 0.5 * p * h) * g_1

    # For N = 2, there is only 1 element
    if N == 2:
        B = np.array([
        (1 + 0.5 * p * h) * g_0 + (1 - 0.5 * p * h) * g_1 + h ** 2 * f(x[1])
    ])

    # Solving for the interior points
    u_interior = np.linalg.solve(A, B)

    # Combine the solution with boundary conditions
    u = np.zeros(N + 1)
    u[0] = g_0
    u[1:-1] = u_interior
    u[-1] = g_1

    # Set up the plot
    plt.figure(figsize=(16, 10))
    t = np.linspace(a, b, 1000)
    plt.plot(t, y(t), label='Exact solution: $u(x) = \\frac{-e^4x+x+e^{2-2x}-e^{2x+2}}{1-e^4}$', color='red', linestyle='--')

    # Plot approximations at grid points
    for i in range(N + 1):
        plt.plot(x[i], u[i], 'o', color='red')
    
    # Specific legend for the approximations
    plt.plot([], [], 'o', color='red', label='Approx. u')

    # Add annotations
    for i in range(N + 1):
        plt.text(x[i], u[i], f'{u[i]:.4f}', ha='right', va='bottom', color='red', fontsize=8)

    # Add labels and title
    plt.title(f"Solution for $u''=4(u - x)$ using Linear Finite Difference Method, for n={N}", fontsize = 20)
    plt.xlabel('x', fontsize = 16)
    plt.ylabel('y', fontsize = 16)
    plt.legend(fontsize = 12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Output matrices A and B
    print("For h =", h, "\n")
    print("Matrix A:\n", A, "\n")
    print("Inverse A:\n", np.linalg.inv(A), "\n")
    print("Vector B:\n", B, "\n")
    print("Solution u:\n", u, "\n")

    # Output the solution at each grid point
    for i in range(0, N + 1):
        print(
            f"u({x[i]:.2f}) = {u[i]:.8f}"
            ", "
            f"Actual value = {y(x[i]):.8f}"
            )
    print("-----------------------------------------------------------------------")

np.set_printoptions(edgeitems=10, linewidth=1000)

# Solving for multiple values of h
for N in [1 / 4, 1 / 8, 1 / 16]:
    bvp_soln(N)