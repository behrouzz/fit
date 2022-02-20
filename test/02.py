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

print(f[peaks])

df_fa = pd.DataFrame({'f':f[peaks], 'a':a[peaks]})
df_fa = df_fa.sort_values('a', ascending=False)
print(df_fa)
#======================================


fig, ax = plt.subplots()
ax.plot(f, a)
ax.scatter(f[peaks], a[peaks], c='r')
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude")
ax.set_xlim(0,0.15)
plt.show()

