import pickle
import numpy as np
import matplotlib.pyplot as plt

data = pickle.load(open("gap.pkl", "rb"))

ogw_mean = np.mean(list(data['ogw'].values()), axis=1)
ogw_std = np.std(list(data['ogw'].values()), axis=1)
omega_mean = np.mean(list(data['omega'].values()), axis=1)
omega_std = np.std(list(data['omega'].values()), axis=1)

xs = list(data['ogw'].keys())
plt.figure(figsize=(4, 3))
# plt.fill_between(xs, np.maximum(0, ogw_mean - ogw_std), ogw_mean + ogw_std, alpha=0.2, color='#377eb8')
plt.plot(xs, ogw_mean, color='#377eb8', label=r"OGW")

# plt.fill_between(xs, np.maximum(0, omega_mean - omega_std), omega_mean + omega_std, alpha=0.2, color='#e41a1c')
plt.plot(xs, omega_mean, color='#e41a1c', label=r"$\Omega$")

plt.xlabel(r"$\delta_g$")
plt.margins(0, 0)
plt.legend()
plt.tight_layout()
plt.savefig("MUTAG_ogw_omega.pdf")
exit()
# plt.ylabel(r"OGW - $\Omega$")
################################################################################
gap = {}
for k in data['ogw']:
    gap[k] = np.array(data['ogw'][k]) - np.array(data['omega'][k])
gap_mean = np.mean(list(gap.values()), axis=1)
gap_std = np.std(list(gap.values()), axis=1)
print(gap_mean)
print(gap_std)
xs = list(data['ogw'].keys())
plt.figure(figsize=(4, 3))
plt.fill_between(xs, np.maximum(0, gap_mean - gap_std), gap_mean + gap_std, alpha=0.2, color='#e41a1c')
plt.plot(xs, gap_mean, color='#e41a1c')
plt.xlabel(r"$\delta_g$")
plt.ylabel(r"OGW - $\Omega$")
plt.margins(0, 0)
# plt.legend()
plt.tight_layout()
plt.savefig("tmp.png")
# print(omega_mean)
# print(omega_std)
