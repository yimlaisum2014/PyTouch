import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

sns.set_theme(style="darkgrid")

folder = '2022-05-01/16-39-29'

df = pd.read_csv('~/PyTouch/train/outputs' +
                 folder+'/train.csv')


smooth_scale = 10000
df.steps = (df.steps//smooth_scale)*smooth_scale / 1e6

# plt.figure(figsize=(15, 10))
sns.lineplot(
    data=df,
    x="steps",
    y='episode_return',
    # hue='total_agent',
    palette=sns.color_palette("flare", as_cmap=True),
)
plt.xlabel('million steps')
plt.savefig(folder + "/episode_return")
plt.show()



df = pd.read_csv('~/subt-virtual/acme_logs/' +
                 folder+'/logs/learner/logs.csv')

sns.lineplot(
    data=df,
    x="steps",
    y='policy_loss',
    # hue='total_agent',
    palette=sns.color_palette("flare", as_cmap=True),
)
plt.xlabel('million steps')
plt.savefig(folder + "/policy_loss")
plt.show()

# plt.close()

