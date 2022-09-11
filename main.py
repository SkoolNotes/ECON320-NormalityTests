# data utils
from contextlib import redirect_stderr
from matplotlib.mathtext import MathtextBackend
import pandas as pd
# plotting utils
import seaborn as sns
sns.set()
# plot
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt, matplotlib.dates as mdates
# import math
import math
# file
from pathlib import Path
# Stats
import numpy as np
import scipy.stats
# sigfigs
from sigfig import round

FILE = "./data/^IXIC.csv"

# Utility functions
def mean_confidence_interval(data, confidence=0.99):
    """calculate confidence interval

    Attributes:
        data (array-like): data
        [confidence] (float): confidence band

    Returns:
        (mean, lower bound, upper bound)
    """
    a = 1.0 * np.array(data.dropna())
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

# in case the price is $0$, for some reason
def safelog(x):
    try:
        return math.log(x)
    except ValueError:
        return math.log(x+1)

# load the file
df:pd.DataFrame = pd.read_csv(FILE)

# calculating continuously componded returns
Rt = df.Close.apply(safelog) - df.Close.shift(1).apply(safelog)
Rt = Rt
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

*Close to Close*
Mean Compounded Return: {round(Rt_mu,3):.2e}
Compound Return Standard Dev.: {round(Rt_std,3):.2e}
1st-Order Autocorrelation: {round(r,3):.2e}
Confidence Interval: {round(mean, 3):.2e}±{round(h, 3):.2e}""")

# get result
fig = sns.histplot(x=Rt, bins=50)
fig.set_title(f"Continuous Return for {Path(FILE).stem}, Close-to-Close")
fig.set_xlabel("Continuous Return")

plt.savefig(f"/Users/houliu/Downloads/results/res_{Path(FILE).stem}_D2C.png")

fig.clear()

#################

rl_std = Rt.rolling(30).std()
rl_mean = Rt.rolling(30).mean()

stability = pd.DataFrame({"date": df.Date,
                          "rolling_std": rl_std,
                          "rolling_mean": rl_mean})

stability.dropna(inplace=True)

stability["loss"] = (rl_std**2 + rl_mean**2)**0.5
sorted_stability = stability.sort_values(["loss"])
stability["date"] = pd.to_datetime(stability["date"])

fig = sns.lineplot(x=stability.date, y=stability.rolling_std)
fig = sns.lineplot(x=stability.date, y=stability.rolling_mean)

plt.legend(labels=['std', 'mean'])
fig.set_title(f"Mean and Std Continuous Return for {Path(FILE).stem}, Over Time")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(5)) 

plt.savefig(f"/Users/houliu/Downloads/results/sorted_{Path(FILE).stem}_D2C.png")

# get the top 1000 most stable day-to-day
reset = sorted_stability.loc[sorted(sorted_stability.iloc[:1000].index)]

reset.date.tolist()

# clear fig
fig.clear()

#################

# calculating continuously componded returns
Rt = df.Close.apply(safelog) - df.Open.apply(safelog)
Rt = Rt
# normalize the returns
Rt_mu = Rt.mean()
Rt_std = Rt.std()
Rt_norm = (Rt-Rt_mu)/Rt_std

# perform autocorrelation
r = (Rt_norm*Rt_norm.shift(1)).sum()/(Rt_norm**2).sum()

# get confidence band
mean, h = mean_confidence_interval(Rt)

print(f"""
*Open to Close*
Mean Compounded Return: {round(Rt_mu,3):.2e}
Compound Return Standard Dev.: {round(Rt_std,3):.2e}
1st-Order Autocorrelation: {round(r,3):.2e}
Confidence Interval: {round(mean, 3):.2e}±{round(h, 3):.2e}""")

# get result
fig = sns.histplot(x=Rt, bins=50)
fig.set_title(f"Continuous Return for {Path(FILE).stem}, Open-to-Close")
fig.set_xlabel("Continuous Return")

plt.savefig(f"/Users/houliu/Downloads/results/res_{Path(FILE).stem}_O2C.png")

fig.clear()
