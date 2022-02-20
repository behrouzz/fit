import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/earth_10yr_365.csv')
df['t'] = pd.to_datetime(df['t'])

x = df['jd'].values
y = df['i'].values

def R2(x, y, func):
    res = y - func(x)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

#======================================
coefs = np.polyfit(x, y, 1)
func = np.poly1d(coefs)
yy = y-func(x)
#======================================
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
fft = np.fft.fft(yy)
N = len(yy)
a = np.abs(fft)[:N//2]
f = np.linspace(0, 1, N)[:N//2]
peaks, _ = find_peaks(a, height=0.1)
df_fa = pd.DataFrame({'f':f[peaks], 'a':a[peaks]})
df_fa = df_fa.sort_values('a', ascending=False)
#======================================

def init_func(t, *args):
    A,w,p = args
    return  A*np.sin(w*t+p)

p0 = [0, 2*np.pi*df_fa['f'].iloc[0], 0]

def fit(t, y):
    popt, pcov = curve_fit(init_func, t, y, maxfev=50000, p0=p0)
    return popt

popt = fit(x, yy)

def final_func(t):
    return init_func(t, *popt)

print('R2 =', R2(x, yy, final_func))
print(popt)

fig, ax = plt.subplots()
ax.plot(x, yy)
ax.plot(x, final_func(x), c='r', alpha=0.5)
plt.show()
