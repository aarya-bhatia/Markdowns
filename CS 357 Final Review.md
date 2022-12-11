# CS 357 Final Review

## Monte Carlo 

Estimating Amount of Clay

* adjust total iterations as neccessary
* np.random.uniform

```{python}
total=int(1e5)
c = 0

for i in range(int(total)):
    x = np.random.uniform(-5,5)
    y = np.random.uniform(-5,5)
    z = np.random.uniform(0,2)
    
    if x**2+y**2<25 and z>0 and z<f(x,y):
        c+=1
        
volume = (c/total)*10*10*2
```

## Floating Point

To convert a positive decimal number x into normalized binary number for given system.

- n = no. of bits in fractional parts
- [-p,p] = exponent range

```python
## |x| = (1.f) x 2^m => log2 |x| = m + log2(1.f) ~ m
def decimal_to_floating_point(x, n, p):
    m = int(np.floor(np.log2(x))) # exponent
    temp = x/2**m - 1 # significand
    f = ''
    # convert float f (0 < f < 1) to binary
    for i in range(n):
        temp = temp*2
        temp_int = int(temp)
        temp = temp - temp_int
        f += str(temp_int)    
    
    return f,m
  

# To convert an integer in deciaml to binary string
def int_to_bin(x):
  f = ''
  while x > 0:
      f += str(x % 2)
      x = x // 2
  return f


# Convert floating point to binary:
x = 10.375
x_int = int(x)
x_frac = x - x_int

# Convert fractional part to binary
b_frac = ''
while True:
    tmp = x_frac * 2
    tmp_int = int(tmp)
    tmp_frac = tmp - tmp_int
    if tmp_frac == 0: 
        b_frac += str(tmp_int)
        break
    x_frac = tmp_frac
    b_frac += str(tmp_int)
    
# Convert integer part to binary
b_int = ''
while x_int > 0:
    b_int += str(x_int % 2)
    x_int = x_int // 2

b = b_int + '.' + b_frac + '000'
print(b)

# strip left zeros
for i in range(len(b)):
    if b[i] == '1':
        b = b[i:]
        break

# strip right zeros
for i in range(len(b)-1,-1,-1):
    if b[i] == '1':
        b = b[:i+1]
        break

print(b)
```

### Rounding Error

```python
L=-5
U=6
n=4
y=2.59375

m = np.math.floor(np.log2(y))
x = y/2**m - 1

assert y == (x+1)*2**m

f = ''
while x>0:
    tmp = x*2
    tmp_int = int(tmp)
    f += str(tmp_int)
    x = tmp-tmp_int
    
print(f)

# Compute by rounding f to n places
rounded = '0100'

float_y = 1

for i in range(len(rounded)):
    float_y += int(rounded[i]) * (2**(-i-1))

float_y *= (2**m)

error = np.abs(float_y-y)
error
```

### Floating point summation

* add negative and positive numbers separately
* add numbers from smallest to biggest

```{python}
neg = []
pos = []

for i in data:
    if i < 0: neg.append(-i)
    elif i > 0: pos.append(i)

sum_neg = 0
sum_pos = 0

for val in sorted(neg):
    sum_neg += val
for val in sorted(pos):
    sum_pos += val

data_sum = sum_pos - sum_neg
```

### Minifloats 

- 9 bit numbers: x = (-1)^s (1.f) 2^m

- Bit 0: Sign bit

- Bit 1-5: Exponent (excess 16 encoding)
- Bit 6-8: Significand (normalized form)

```{python}
import numpy as np

def minifloat_to_decimal(x):
    S = 1-2*x[0]
    E = x[1]*16+x[2]*8+x[3]*4+x[4]*2+x[5]*1
    M = 2 ** int(E - 16)
    F = 1 + x[6]/2 + x[7]/4 + x[8]/8
    return S * F * M

def sum_minifloats(x,y):
    float_x = minifloat_to_decimal(x)
    float_y = minifloat_to_decimal(y)
    z = float_x + float_y
    return z
```

### Estimating Digits

```{python}
import numpy as np
import numpy.linalg as la
import math

t = np.abs(np.log10(la.cond(A)))
s = np.abs(np.log10(np.finfo(float).eps))
correct_digits = math.floor(s-t)

print(s, t, s-t, correct_digits)

x = la.solve(A,b)
```



## PCA Algorithm

* A = Data Set = m x n, m = features, n = samples
* To reduce A to Astar = m x k without loss of information
* Step 1: Shift data set A to have zero mean: A = A - A.mean()
* Step 2: Compute SVD: A = U, S, Vt
* Variance = Singular Values ^ 2
* Principal directions of data set = Columns of V
* Step 3: New data set: Astar = A . V = U . Sigma
* Step 3: To reduce dimension of data set, we use first k columns of V only: Astar = A . V[:, :k] = A . Vt[:k, :]. Astar has dim = m x k.

Q: Number of principal components needed to capture x % of variance:

```{python}
# U,S,Vt = svd(A)
variance = S**2
np.cumsum(variance)/variance.sum() * 100
```



## SVD

### To solve a system Ax = b with SVD:

* np.linalg.pinv: pseduoinverse
* x = pinv(A) @ b = V @ pinv(S) @ U.T @ b

```{python}
U, S, Vh = np.linalg.svd(A)

Si = np.zeros((Vh.T.shape[1], U.T.shape[0]))

for idx, val in enumerate(S):
    if val > 1e-10:
        Si[idx,idx] = 1/val

x = Vh.T @ Si @ U.T @ b
```

### Approximating Physical Constants

```{python}
import numpy as np
import numpy.linalg as la

## ln K = ln A - E/RT
## y(T) = C1 + C2/T
## A = exp(C1)
## E = -R * C2

k = data[:,0]
T = data[:,1]

N = data.shape[0]
A = np.vstack([np.ones(N), 1/T]).T

c1, c2 = la.lstsq(A, np.log(k), rcond=None)[0]

A = np.exp(c1)
E = -10.51 * c2
coef = np.array([A,E])
k0 = np.exp(c1 + c2/79)
```

### Image Compression

```{python}
U, S, Vt = la.svd(image, full_matrices = False) # reduced svd

## k = argmin || A - A_k || <= eps
## || A - A_k || = (k+1)st singular value
for i, s in enumerate(S):
    if s <= eps:
        k = i
        break

img_k = np.zeros(image.shape)

for i in range(k):
    img_k += S[i] * np.outer(U[:,i], Vt[i,:])

image_compressed = img_k
```



## Markov Chains

```{python}
import numpy as np
import numpy.linalg as la

N = len(genres)
T = np.diag(np.ones(N) * 0.60) # Transition Matrix

for u in range(N):
    colsum = np.sum(A[:,u])
    
    if colsum == 0: 
        T[:,u] = 0.40/N
        continue
    
    for adj in range(N):
    		if A[adj][u]: T[adj][u] = 0.40/colsum

print(T)

x = np.ones(N)
x /= la.norm(x,1)

tol = 1e-15

while 1:
    x1 = T @ x
    x1 /= la.norm(x1, 1)
    
    if la.norm(x1-x, 2) < tol:
        break
    
    x = x1

P = x[genres.index("Soul")] # Steady state probability
print(P)
```

- **Google Matrix**: `G = alpha * A + ((1-alpha)/N) * np.ones((N,N))` 



## CSR to Dense Matrix

```python
import numpy as np

## Given: A_csr: { shape, indices, indptr, data }

A = np.zeros(A_csr.shape)
off = 0
prevptr = 0
row = 0


for rowptr in A_csr.indptr[1:]:
    num = rowptr -  prevptr
    for j in range(num):
        col = A_csr.indices[off+j]
        A[row][col] = A_csr.data[off + j]
    
    off += num
    row += 1
    prevptr = rowptr
```



## Taylor Series

Q: ` f(x) = sin(x), a = 0, x = 2, deg = 4`

```{python}
derivatives = [0, 1, 0, -1] # ( sin -> cos -> -sin -> -cos ) -> ( sin -> cos .... )
ans = 0

for i in range(5):
    p = 2**i # x^n
    fact = math.factorial(i) # n!
    der = derivatives[i % 4] # nth derivative of f at x=0
    ans += p*der / fact

ans # 0.6666666666666667
```



## Steepest Descent

```{python}
import sympy as sp

f=14*x**2+6*x*y+14*y**2+8*sp.sin(y)**2+6*sp.cos(x*y)
gx = sp.diff(f,x).subs({x: -1, y: 3}).evalf()
gy = sp.diff(f,y).subs({x: -1, y: 3}).evalf()
-1 - 0.05 * gx, 3 - 0.05 * gy
```



## Bisection Method 

How much wood is needed?

Note: **np.sign**

```{python}
import numpy as np

"""
Solve T(w) = 375 => T(w) - 375 = 0
Let: f(x) = T(x) - 375 => Solve, f(x) = 0.
"""

def f(w):
    return get_temperature(w) - 375

a = 0
b = max_wood
fa = f(a)
fb = f(b)
intervals = []

for i in range(1000):
    intervals.append((a,b))
    m = (a+b)/2
    fm = f(m)

    if abs(fm) < epsilon:
        weight = m
        break
    
    if np.sign(fa) != np.sign(fm):
        b = m
    else:
        a = m
```



## Linear Systems

An algorithm to perform proofs

```python
import numpy as np

A = []
algorithm_list = ['Lucky_Tosser', 'Math_Hater', 'Problem_Breaker', 'Useless_Loser', 'Random_Guesser']

for experiment in experiment_data:
    data = dict(experiment_data[experiment])
    row = []
    for method in algorithm_list:
        if method in data:
            row.append(data[method])
        else:
            row.append(0)
    
    A.append(row)
    
A = np.array(A)
print(A)


# num proofs for each method
percent_attempt = np.linalg.solve(A, np.array(proved))
```



## Optimization

Optimization ND Newton Method one step

```python
# Optimization with Newton Method in 2d

# GIVEN FUNCTION f(x,y)
def func(w):
    x,y=w
    return 5*x**2+2*y**4

# GRADIENT VECTOR G = [dx,dy]
def grad(w):
    x,y = w
    return np.array([
        10*x,
        8*y**3
    ])

# HESSIAN MATRIX H = [[dxx,dxy],[dyx,dyy]]
def hess(w):
    x,y=w
    
    return np.array([
        [10, 0],
        [0, 24*y**2]
    ])

# INITIAL X
w0 = np.array([2,3])

# UPDATE STEP
w1 = w0 - np.linalg.solve(hess(w0), grad(w0))

print(w1)
```

