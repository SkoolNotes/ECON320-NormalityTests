# data utils
from contextlib import redirect_stderr
import pandas as pd
# plotting utils
import seaborn as sns
sns.set()
# plot
import matplotlib.pyplot as plt
# import math
import math
# file
from pathlib import Path
# Stats
import numpy as np
import scipy.stats
# sigfigs
from sigfig import round

FILE = "./data/^RUT.csv"

# Utility functions
def mean_confidence_interval(data, confidence=0.99):
    """calculate confidence interval

    Attributes:
        data (array-like): data
        [confidence] (float): confidence band

    Returns:
        (mean, lower bound, upper bound)
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

# load the file
df:pd.DataFrame = pd.read_csv(FILE)

# calculating continuously componded returns
Rt = df.Close.apply(math.log) - df.Close.shift(1).apply(math.log)
Rt = Rt.dropna()
# normalize the returns
Rt_mu = Rt.mean()
Rt_std = Rt.std()
Rt_norm = (Rt-Rt_mu)/Rt_std

# perform autocorrelation
r = (Rt_norm*Rt_norm.shift(1)).sum()/(Rt_norm**2).sum()

# get confidence band
mean, h = mean_confidence_interval(Rt)

print(f"""
Ticker: {Path(FILE).stem}
Mean Compounded Return: {round(Rt_mu,3):.2e}
Compound Return Standard Dev.: {round(Rt_std,3):.2e}
1st-Order Autocorrelation: {round(r,3):.2e}
Confidence Interval: {round(mean, 3):.2e}±{round(h, 3):.2e}""")

# get result
fig = sns.histplot(x=Rt, bins=50)
fig.set_title(f"Continuous Return for {Path(FILE).stem}")
fig.set_xlabel("Continuous Return")

plt.savefig(f"/Users/houliu/Downloads/res_{Path(FILE).stem}.png")

fig.clear()


