import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('test.csv')
df['t'] = pd.to_datetime(df['t'])

x = df['jd'].values
y = df['i_cont'].values

def R2(x, y, func):
    res = y - func(x)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

#======================================
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
fft = np.fft.fft(y)
N = len(y)
a = np.abs(fft)[:N//2]          # Amplitude
f = np.linspace(0, 1, N)[:N//2] # Frequency
peaks, _ = find_peaks(a, height=0.1)
df_fa = pd.DataFrame({'f':f[peaks], 'a':a[peaks]})
df_fa = df_fa.sort_values('a', ascending=False)

#======================================


[0, 2*np.pi*df_fa['f'].iloc[0], 0]
[0, 2*np.pi*df_fa['f'].iloc[0], 0] + [0, 2*np.pi*df_fa['f'].iloc[1], 0]
[0, 2*np.pi*df_fa['f'].iloc[0], 0] + [0, 2*np.pi*df_fa['f'].iloc[1], 0] + [0, 2*np.pi*df_fa['f'].iloc[2], 0]

p0 = []
for i in range(5):

    def init_func(t, *args):
        A = args[0::3]
        w = args[1::3]
        p = args[2::3]
        return np.array([A[i]*np.sin(w[i]*t+p[i]) for i in range(len(A))]).sum()

    p0 = [*p0, *[0, 2*np.pi*df_fa['f'].iloc[i], 0]]
    
    def fit(tt, yy):
        popt, pcov = curve_fit(init_func, tt, yy, p0=p0)
        return popt

    popt = fit(x, y)

    def final_func(t):
        return init_func(t, *popt)

    print('R2 =', R2(x, y, final_func))
    #print(p0)
