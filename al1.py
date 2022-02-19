import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/venus.csv')
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

from scipy.optimize import curve_fit

def init_func(t, *args):
    c,A,w,p = args
    return  c + A*np.sin(w*t+p)

fft = np.fft.fft(y)
N = len(y)

a = np.abs(fft)[:N//2][1:]          # Amplitude
f = np.linspace(0, 1, N)[:N//2][1:] # Frequency

df_fa = pd.DataFrame({'f':f, 'a':a})
df_fa = df_fa.sort_values('a', ascending=False)
#print(df_fa)

A_,w_,p_ = [ 5.13776580e-02,  4.45507691e-05,  1.30251470e+03]

def fit(tt, yy):
    p0 = [0, A_,w_,p_]
    popt, pcov = curve_fit(init_func, tt, yy, p0=p0)
    print('1111')
    return popt

popt = fit(x, y)
print(popt)

def final_func(t):
    return init_func(t, *popt)

print('R2 =', R2(x, y, final_func))

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, final_func(x), c='r', alpha=0.5)
ax.ticklabel_format(style='plain')
plt.show()


