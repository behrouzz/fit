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
print('Coefs:', coefs)
print('R2 =', R2(x, y, func))

fig, ax = plt.subplots()
ax.plot(x, y)
#ax.plot(x, func(x), c='r', alpha=0.5)
ax.plot(x, coefs[0]*x + coefs[1], c='r', alpha=0.5)
ax.plot(x, y-func(x), c='g', alpha=0.5)
ax.ticklabel_format(style='plain')
plt.show()


