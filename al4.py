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

fft = np.fft.fft(y)
N = len(y)

a = np.abs(fft)[:N//2][1:]          # Amplitude
f = np.linspace(0, 1, N)[:N//2][1:] # Frequency

df_fa = pd.DataFrame({'f':f, 'a':a})
df_fa = df_fa.sort_values('a', ascending=False)

#---------------------------------------------------

g = [ 3.71819958e+00,  3.23813485e-01, -1.78907815e-05,  1.45617183e+03]
#y_ = g[0] + g[1]*np.sin(g[2]*t+g[3])
p0 = [0, 2*np.pi*df_fa['f'].iloc[1], 0]
for nn in range(5):

    def init_func(t, *m):
        global y_, m0
        m = np.array(m)
        A = m[0::3]
        w = m[1::3]
        p = m[2::3]
        for i in range(len(A)):
            y_ = g[0]+g[1]*np.sin(g[2]*t+g[3]) + A[i]*np.sin(w[i]*t+p[i])
        return  y_
    
    def fit(tt, yy):
        popt, pcov = curve_fit(init_func, tt, yy, p0=p0)
        return popt

    popt = fit(x, y)

    def final_func(t):
        return init_func(t, *popt)

    #print('R2 =', R2(x, y, final_func))

    p0 = list(popt) + [0, 2*np.pi*df_fa['f'].iloc[nn+2], 0]
    p0 = np.array(p0)
    print(p0.shape)
    """
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.plot(x, final_func(x), c='r', alpha=0.5)
    ax.ticklabel_format(style='plain')
    ax.set_title('R2 ='+str(R2(x, y, final_func)))
    plt.show()
    """
    


