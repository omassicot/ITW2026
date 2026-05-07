from functools import lru_cache
from itertools import product
from math import prod
from collections import defaultdict
from mpmath import sqrt, log, exp, floor, ceil, mpf, mp, re, nstr
from scipy.special import ndtri
import pandas as pd

mp.dps = 100  # set decimal places for mpmath
plotting_digits = 4  # number of digits to display in the final plot data

### Problem data
P = ((mpf(1)/3,mpf(1)/3),(mpf(1)/3,0))
epsilon = mpf('1e-1')
ns = list(range(2, 600+1))
ds = [sqrt(log(n)/n)/12 for n in ns]
nfordelta = [10, 20, 40, 100, 200, 400]
verbose = True

### To compute Lambda with caching
@lru_cache(maxsize=None)
def factorial(n):
    return 1 if n <= 0 else n * factorial(n-1)

def LambdaU(T):
    return factorial(sum(T)) * prod(p**k/factorial(k) for p, k in zip(PU, T))

@lru_cache(maxsize=None)
def LambdaV(T):
    return factorial(sum(T)) * prod(p**k/factorial(k) for p, k in zip(PV, T))

### To generate all types
def flat_types(n,d):
    lower_bounds = tuple(max(0,int(ceil(n*p-n*d))) if p > 0 else 0 for p in fP)
    upper_bounds = tuple(int(floor(n*p+n*d)) if p > 0 else 0 for p in fP)
    potential_lower_sums = tuple(sum(lower_bounds[i:]) for i in range(len(fP)+1))
    potential_upper_sums = tuple(sum(upper_bounds[i:]) for i in range(len(fP)+1))
    types = set()
    current_type = tuple()
    current_sum = 0
    def backtrack(i):
        nonlocal current_type, current_sum, types
        minimum = max(lower_bounds[i], n - potential_upper_sums[i+1] - current_sum)
        maximum = min(upper_bounds[i], n - potential_lower_sums[i+1] - current_sum)
        if i == len(fP) - 1:
            types.add(tuple(current_type + (minimum,)))
            return
        for count in range(minimum, maximum + 1):
            current_type += (count,)
            current_sum += count
            backtrack(i + 1)
            current_type = current_type[:-1]
            current_sum -= count
    backtrack(0)
    return types

### Convenient probabilities
def flatten(o):
    return tuple(a for b in o for a in b)

def reshape(p):
    return tuple(p[i*V:(i+1)*V] for i in range(U))

fP = flatten(P)
piUV = min(p for p in fP if p > 0)
PU = tuple(sum(probs) for probs in P)
piU = min(p for p in PU if p > 0)
U = len(PU)
PV = tuple(sum(probs) for probs in zip(*P))
piV = min(p for p in PV if p > 0)
V = len(PV)
PUPV = tuple(p*q for p, q in product(PU, PV))
PVgU = tuple(tuple(q/p for q in PuV) for p,PuV in zip(PU,P))

### Asymptotics
I = sum(x*log(x/y) for x,y in zip(fP, PUPV) if x > 0)/log(2)
Var = sum(p*(sum(x*log(x/y) if x>0 else 0 for x,y in zip(PVgu, PV))-I)**2 for p,PVgu in zip(PU, PVgU))/log(2)**2
print(f"I(U;V) = {I:.4f}, Var = {Var:.4f}")

### For each n, find R
Rs = []
for n,d in zip(ns,ds):
    Rs += [float('inf')]
    types = {reshape(t) for t in flat_types(n,d)}
    types_by_marg = defaultdict(list)
    for T in types:
        types_by_marg[tuple(sum(row) for row in T)].append(T)

    if not any(T for T in types_by_marg.values()):
        if verbose:
            print(f"n={n}, \tR=inf, \t\tdelta={d:.4f}")
        continue
    
    Pis = {S:sum(prod(LambdaV(t) for t in T) for T in types_by_marg[S]) for S in types_by_marg}
    Lus = {S:LambdaU(S) for S in types_by_marg}
    Pi_to_Lu = defaultdict(mpf)
    for k, v1 in Pis.items():
        if k in Lus:
            Pi_to_Lu[v1] += Lus[k]

    def val(m):
        # Using exp(m*log(1-p)) ~ exp(-m*p) 
        # for very small p for numerical stability
        return re(sum(
            lu*exp(m*(log(1-p) if p > 1e-15 else -p))
            for p, lu in Pi_to_Lu.items())) 

    if val(1) <= epsilon:
        if verbose:
            print(f"n={n}, \tR=0, \tdelta={d:.4f}")
        Rs[-1] = 0
        continue
    
    R = I*log(2)
    while val(exp(n*R)) > epsilon:
        R *= 2
    r = mpf(0)
    while R-r > mpf(f"1e-{plotting_digits}")*I:
    # Dichotomy over R, with plotting precision
        rm = (R+r)/2
        if val(exp(n*rm)) < epsilon:
            R = rm
        else:
            r = rm
    # Adjusting R to match an integer number of messages
    # when using R real instead of log(M)/n is too bad,
    # i.e., when Lemma 4 is insufficiently accurate.
    if exp(-n*R)/n > mpf(f"1e-{plotting_digits}")*I:
        M = ceil(exp(n*R))
        m = floor(exp(n*r))
        while M > m + 1:
            mm = (M + m) // 2
            if val(mm) < epsilon:
                M = mm
            else:
                m = mm
        R = log(M)/n
    if verbose:
        print(f"n={n}, \tR={R:.6f}, \tdelta={d:.4f}")
    Rs[-1] = R/log(2)    

qinve = ndtri(float(1-epsilon))
Rapprox = [I+sqrt(Var/n)*qinve for n in ns]

def sig(x, digits=plotting_digits):
    return float(nstr(x, digits))
pd.DataFrame({
    "n": ns,
    "R": [sig(r) for r in Rs],
    "d": [sig(d) for d in ds],
    "I": [sig(I)]*len(ns),
    "Rapprox": [sig(r) for r in Rapprox]
}).to_csv('plot_data.csv', index=False)

deltas = [d for n, d in zip(ns, ds) if n in nfordelta]
print(r"\begin{tabular}{c|" + "c"*len(deltas) + "}")
print("$n$      & " + " & ".join(str(n) for n in nfordelta) + r" \\")
print(r"\hline")
print(r"$\delta$ & " + " & ".join(f"{d:.2g}" for d in deltas) + r" \\")
print(r"\end{tabular}")