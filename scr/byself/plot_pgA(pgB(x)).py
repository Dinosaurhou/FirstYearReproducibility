# Python code to compute and plot Fig.2 from the Supplementary Information
# of "Catastrophic cascade of failures in interdependent networks".
# This reproduces the panel showing y = p * gA[p * gB(x)] and y = x for the
# scale-free distribution used in the SI: PA(k)=PB(k) = (2/k)^2 - (2/(k+1))^2 for k>=2.
# The code also shows the iteration arrows starting at x = p'0 = p.
# Follows the analytic expressions in the Supplementary Information.
import numpy as np
import matplotlib.pyplot as plt
from math import isclose

# --- degree distribution (as used in SI) ---
k_min = 2
k_max = 2000  # truncation for practical sums; increase if needed
ks = np.arange(k_min, k_max+1)
PA = (2.0/ks)**2 - (2.0/(ks+1))**2
# PA = PA / PA.sum()  # normalize to be safe (should already sum ~1 over k>=2)

# precompute moments if needed
k_mean = (ks * PA).sum()

# generating functions GA0(z) and GA1(z)
def GA0(z):
    # sum PA(k) z^k
    return np.sum(PA * (z**ks))

def GA0_prime(z):
    return np.sum(PA * ks * (z**(ks-1)))

def GA1(z):
    # GA1(z) = GA0'(z) / GA0'(1)
    denom = GA0_prime(1.0)
    return GA0_prime(z) / denom

# solve for f given p and network's GA1 via fixed-point iteration:
def solve_f(p, GA1_func, tol=1e-12, maxiter=2000):
    # solve f = GA1(1 - p(1 - f))
    f = 0.5  # initial guess
    for i in range(maxiter):
        arg = 1.0 - p * (1.0 - f)
        f_new = GA1_func(arg)
        if abs(f_new - f) < tol:
            return max(0.0, min(1.0, f_new))
        f = f_new
    # fallback
    return f

# compute g(p) = 1 - GA0(z) with z = 1 - p(1 - f) (as in SI)
def compute_g(p):
    f = solve_f(p, GA1)
    z = 1.0 - p * (1.0 - f)
    return 1.0 - GA0(z)

# For a given p, define H(x) = p * gA(p * gB(x))
# For symmetric case gA = gB = g using same distribution.
def H_of_x(x, p):
    # first compute gB(x) for argument x (here x is fraction of nodes -> treat as µ)
    # The function compute_g expects a p-like argument in [0,1]. We'll treat x as that fraction.
    gB_x = compute_g(x)
    inner = p * gB_x
    gA_inner = compute_g(inner)
    return p * gA_inner

# Prepare x grid and p values as in SI figure
xs = np.linspace(0.0, 1.0, 401)
ps = [0.70, 0.752, 0.80]

# Compute curves
curves = {p: np.array([H_of_x(x, p) for x in xs]) for p in ps}

# Plotting: one figure showing y=x and the three H curves + iteration arrows for p ~ pc
plt.figure(figsize=(6,6))
plt.plot(xs, xs, label='y = x')  # line
for p in ps:
    plt.plot(xs, curves[p], label=f'p = {p}')

# iteration arrows: show iteration path starting at x = p (i.e., mu'_0 = p)
def iteration_path(p, steps=12):
    pts = []
    x = p  # starting x = p'0
    pts.append(x)
    for _ in range(steps):
        x = H_of_x(x, p)  # next x is H(x)
        pts.append(x)
    return np.array(pts)

# draw iterations for p = 0.752 (near critical) as arrows
p_iter = 0.752
pts = iteration_path(p_iter, steps=12)
# plot staircase iteration: vertical then horizontal segments
for i in range(len(pts)-1):
    x0 = pts[i]
    x1 = pts[i+1]
    # vertical: (x0, x0) to (x0, H(x0)=x1)
    plt.plot([x0, x0], [x0, x1], linestyle='--', linewidth=0.8)
    # horizontal: (x0, x1) to (x1, x1)
    plt.plot([x0, x1], [x1, x1], linestyle='--', linewidth=0.8)
# mark iteration points
plt.scatter(pts, pts, s=10)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Reproduction of SI Fig.2: y = pgA[pgB(x)] and y=x (SF λ=3)')
plt.xlim(0,1)
plt.ylim(0,1)
plt.legend()
plt.grid(True)
plt.show()
