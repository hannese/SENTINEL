import scipy.stats
import numpy as np

def create_distribution(masses, boundary):
    if len(masses) == len(boundary):
        num_masses = len(masses)
        z_min = boundary[0]
        z_max = boundary[-1]
        new_boundary = np.linspace(z_min, z_max, num=num_masses+1)
        hist = [masses, new_boundary]
    else:
        hist = [masses, boundary]

    return scipy.stats.rv_histogram(hist)

def wang(masses, boundary, alpha):
    masses = np.array(masses)
    boundary = np.array(boundary)
    dist = create_distribution(masses, boundary)
    if np.isclose(boundary[0], boundary[-1]):
        return masses.T @ boundary
    dwang = lambda t : scipy.stats.norm.pdf(
        scipy.stats.norm.ppf(t) + alpha) / (scipy.stats.norm.pdf(
        scipy.stats.norm.ppf(t)) + 1e-8)

    taus = (boundary - boundary[0]) / (boundary[-1] - boundary[0])
    dwangs = np.array([dwang((taus[t]+taus[t+1])/2) for t in range(len(boundary)-1)]).reshape(-1, 1)
    masses_ = np.array([dist.pdf((boundary[t]+boundary[t+1])/2) for t in range(len(boundary)-1)]).reshape(-1, 1)
    masses_ /= np.sum(masses_)
    atoms = np.array(boundary[:-1]).reshape(-1, 1)
    E = np.sum(np.multiply(masses_, np.multiply(atoms, dwangs)))
    if np.isnan(E) or np.isinf(E):
        print("E is inf or NaN")
        return boundary[0]
        #raise RuntimeError("E is inf or NaN", masses_)
    return E

def _Entropic_RM(z, X, alpha=0.95):

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    value = np.mean(np.exp(-1 / z * a), axis=0)
    value = z * (np.log(value) + np.log(1 / alpha))
    value = np.array(value).item()

    return value

from scipy.optimize import minimize
from scipy.optimize import Bounds
def evar(masses, boundary, alpha=0.95):
    masses = np.array(masses)
    boundary = np.array(boundary)
    dist = create_distribution(masses, boundary)
    X = dist.rvs(size=10000)
    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    bnd = Bounds([1e-12], boundary[-1])
    result = minimize(
        _Entropic_RM, [1], args=(X, alpha), method="SLSQP", bounds=bnd, tol=1e-12
    )
    t = result.x
    t = t.item()
    value = -_Entropic_RM(t, X, alpha)
    if np.isnan(value) or np.isinf(value):
        print("E is inf or NaN")
        return boundary[0]
        #raise RuntimeError("Evar value is inf", value, t)
    return value

def cvar(masses, boundary, alpha):
    masses = np.array(masses)
    boundary = np.array(boundary)
    dist = create_distribution(masses, boundary)
    var = dist.ppf(alpha)
    mask = np.where(var >= boundary)
    if len(mask[0]) == len(boundary):
        return dist.mean()
    tmp = dist.pdf(boundary[mask])
    E = np.dot(boundary[mask], tmp) / np.sum(tmp) + 1/alpha * var * (alpha - dist.cdf(var))
    if np.isnan(E) or np.isinf(E):
        print("E is inf or NaN")
        return boundary[0]
        #raise RuntimeError("E is inf or NaN", tmp)
    return E

def std(masses, boundary, c):
    masses = np.array(masses)
    boundary = np.array(boundary)
    dist = create_distribution(masses, boundary)
    return dist.mean() - dist.std() * c

def cvar_():

    alpha = 0.25
    loc = 10
    scale = 1.5
    data = scipy.stats.norm.rvs(size=100000, loc=loc, scale=scale, random_state=123)
    hist = np.histogram(data, bins=100)
    val = cvar(hist[0], hist[1], alpha)
    exact_val = loc - scale * scipy.stats.norm.pdf(scipy.stats.norm.ppf(alpha)) / alpha
    print("empirical: ", val)
    print("exact: ", exact_val)

def wang_():

    alpha = 0.25
    loc = 10
    scale = 1.5
    data = scipy.stats.norm.rvs(size=100000, loc=loc, scale=scale, random_state=123)
    hist = np.histogram(data, bins=100)
    val = wang(hist[0], hist[1], alpha)
    exact_val = loc - scale * alpha
    print("empirical: ", val)
    print("exact: ", exact_val)

def evar_():

    alpha = 0.5
    loc = 10
    scale = 3.5
    data = scipy.stats.norm.rvs(size=100000, loc=loc, scale=scale, random_state=123)
    hist = np.histogram(data, bins=100)
    val = evar(hist[0], hist[1], alpha)
    exact_val = loc - scale * np.sqrt(-2 * np.log(alpha))
    print("empirical: ", val)
    print("exact: ", exact_val)

def cvar_mix():
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    A = np.random.normal(-1, 0.5, size=1000)
    B = np.random.normal(1, 1, size=1000)

    x = np.linspace(-4, 4, 1000)

    px = np.empty((1000, 1000))

    pA = norm.pdf(x, -1, 0.5)
    pB = norm.pdf(x, 1, 1)

    beta_a = 2
    beta_b = 5

    alpha = 0.1

    w = np.random.beta(2, 5, 1000)

    w_ = 1-w
    for i in range(len(w)):
        px[i, :] = pA * w[i] + pB * w_[i]
    val = cvar(np.mean(px, axis=0), x, alpha)
    composite_vals = np.empty((1000))
    for i in range(len(w)):
        composite_vals[i] = cvar(px[i, :], x, alpha)


    hist = np.histogram(composite_vals, bins=100)
    cval = cvar(hist[0], hist[1], alpha)

    plt.plot(x, np.mean(px, axis=0), color='black', linewidth=2)
    plt.ylim([-0.1, 0.4])
    plt.savefig("1_marginal.pdf")
    plt.axis('off')
    plt.savefig("1_marginal_no_axis.pdf")

    # epistemic

    ECE = beta_a / (beta_a+beta_b) * -1 + beta_b / (beta_a+beta_b) * 1
    EP = -1
    AL = beta_a / (beta_a+beta_b) * (-1 - 0.5 * norm.pdf(norm.ppf(alpha)) / alpha) + beta_b / (beta_a+beta_b) * (1 - 1 * norm.pdf(norm.ppf(alpha)) / alpha)
    plt.vlines(EP, 0, 0.265, color='black', linewidth=2)
    plt.text(EP - 0.15, -0.025, r'EP')
    plt.vlines(ECE, 0, 0.245, color='black', linewidth=2)
    plt.text(ECE - 0.25, -0.025, r'$\mathbb{E}\circ\mathbb{E}$')
    plt.vlines(AL, 0, 0.255, color='black', linewidth=2)
    plt.text(AL-0.075, -0.025, r'AL')
    plt.savefig("2_marginal_no_axis.pdf")
    plt.axis('on')
    plt.savefig("2_marginal.pdf")

    plt.clf()

    val = cvar(np.mean(px, axis=0), x, alpha)
    plt.plot(x, np.mean(px, axis=0), color='black', linewidth=2)
    plt.ylim([-0.1, 0.4])
    plt.vlines(val, 0, 0.235, color='black', linewidth=2)
    plt.text(val-0.2, -0.05, r'E$+$A')
    plt.vlines(cval, 0, 0.175, color='black', linewidth=2)
    plt.text(cval-0.2, -0.025, r'E$\circ$A')
    plt.vlines(ECE, 0, 0.245, color='black', linewidth=2)
    plt.text(ECE - 0.25, -0.025, r'$\mathbb{E}\circ\mathbb{E}$')
    plt.savefig("3_marginal.pdf")
    plt.axis('off')
    plt.savefig("3_marginal_no_axis.pdf")
    plt.vlines(EP, 0, 0.265, color='black', linewidth=2)
    plt.text(EP - 0.15, -0.025, r'EP')
    plt.vlines(ECE, 0, 0.245, color='black', linewidth=2)
    plt.text(ECE - 0.25, -0.025, r'$\mathbb{E}\circ\mathbb{E}$')
    plt.vlines(AL, 0, 0.255, color='black', linewidth=2)
    plt.text(AL-0.075, -0.025, r'AL')
    plt.savefig("4_marginal_no_axis.pdf")
    plt.axis('on')
    plt.savefig("4_marginal.pdf")

    plt.plot(x, pA, color='red', linewidth=2)
    plt.plot(x, pB, color='blue', linewidth=2)
    plt.savefig("5_marginal.pdf")
    plt.axis('off')
    plt.savefig("5_marginal_no_axis.pdf")



def main():
    cvar_mix()
    evar_()
    cvar_()
    wang_()

if __name__ == "__main__":
    main()