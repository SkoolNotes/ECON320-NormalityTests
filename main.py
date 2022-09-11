# data utils
from contextlib import redirect_stderr
import pandas as pd
# plotting utils
import seaborn as sns
# plot
import matplotlib.pyplot as plt
# import math
import math
# file
from pathlib import Path

FILE = "./data/AAPL.csv"

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

print(f"""
Stock: {Path(FILE).stem}
Mean Compounded Return: {round(Rt_mu,3)}
Compound Return Standard Dev.: {round(Rt_std,3)}
1st-Order Autocorrelation: {round(r,3)}""")

