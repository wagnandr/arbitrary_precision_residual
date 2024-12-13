import decimal
import numpy as np
import scipy.sparse as sps
import time
import hpresidual as hpr
from matplotlib import pyplot as plt

# n = 5000
#n = 200000
n = 20000
prec = 128 

hpr.set_precision(prec)

def include_boundary_conditions(A):
    A[0, 1] = 0; A[0, 0] = 1
    A[-1, -2] = 0; A[-1, -1] = 1

line = ['-', '--', ':' ]

dtype = np.float64
h = 1./(n-1)**2
A = sps.diags([np.ones(n)*(2/h), -np.ones(n-1)/h, -np.ones(n-1)/h], [0, 1, -1], shape=(n, n), dtype=dtype)
A = A.tocsr()
include_boundary_conditions(A)

x = np.array([dtype(np.sin(i/(n-1) * np.pi)) for i in range(n)], dtype=dtype)

start = time.time()
A2 = A@A
end = time.time()
print(f'matrix multiplication elapsed {end-start}')
# zz = A @ (A @ x)
start = time.time()
zz = -(A2) @ x
end = time.time()
print(f'multiplication elapsed {end-start}')
zz /= np.linalg.norm(zz)

decimal.getcontext().prec = prec

def cos(x):
    decimal.getcontext().prec += 2
    i, lasts, s, fact, num, sign = 0, 0, 1, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i-1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    decimal.getcontext().prec -= 2
    return +s

def sin(x):
    decimal.getcontext().prec += 2
    i, lasts, s, fact, num, sign = 1, 0, x, 1, x, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i-1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    decimal.getcontext().prec -= 2
    return +s
 
dpi = decimal.Decimal('3.141592653589793238462643383279502884197169399375105820974944592307816406286')

start = time.time()
x = [str(sin(decimal.Decimal(i)/decimal.Decimal(n-1) * dpi)) for i in range(n)]
#x = [str(ix) for ix in x]
end = time.time()
print(f'prepare x elapsed {end - start}')

start = time.time()
hprcalc = hpr.create(A2)
end = time.time()
print(f'convert A elapsed {end - start}')

b = np.zeros(A.shape[0])

start = time.time()
hprcalc.set_b(b)
hprcalc.set_x(x)
end = time.time()
print(f'preprocessing elapsed {end - start}')
xx = np.zeros(A.shape[0])
start = time.time()
hprcalc.evaluate(xx)
xx /= np.linalg.norm(xx)
end = time.time()
print(f'multiplication elapsed {end - start}')

plt.plot(zz, '-', label=f'old')
plt.plot(xx, ':', label=f'new')

plt.legend()
plt.grid(True)
plt.show()