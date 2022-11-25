import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

results = [
    "FedProx_res_mu_0_non_iid_True_var_E_False",
    "FedProx_res_mu_0.3_non_iid_True_var_E_True"
]

colors = ["blue", "red"]
legends = []

for color, res in zip(colors, results):
    with open(res+".npy", "rb") as file:
        mean = np.load(file)
        ci = np.load(file)
        x_vec = np.arange(len(mean))
        plt.plot(x_vec, mean, color=color)
        plt.fill_between(x_vec, mean-0.5*ci, mean+0.5*ci, color=color, alpha=0.3)

        legends.append(res)
        legends.append(res+"_CI")

plt.legend(legends)
plt.show()