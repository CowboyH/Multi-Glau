"""
This script provides example visualizations including violin plots, pie charts,
and KDE (Kernel Density Estimation) plots.

NOTE:
- All data used in this script are randomly generated and do not correspond to any
  actual study results.
- The figures are for demonstration purposes only and should not be interpreted
  as scientific findings.

You may modify the data and visualization parameters based on your specific needs.
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Violin Plot: MD distribution across disease stages
# -------------------------------

# Simulate MD data for each stage
np.random.seed(0)
stages = ['Early', 'Moderate', 'Advanced', 'Severe']
md_values = {
    'Early': np.random.normal(loc=-2, scale=2, size=100),
    'Moderate': np.random.normal(loc=-10, scale=2, size=100),
    'Advanced': np.random.normal(loc=-20, scale=2, size=100),
    'Severe': np.random.normal(loc=-30, scale=3, size=100)
}

# Create a DataFrame
df_md = pd.DataFrame([(stage, value) for stage, values in md_values.items() for value in values],
                     columns=['Stage', 'MD'])

# Draw the violin plot with overlaid strip plot
plt.figure(figsize=(6, 4))
sns.violinplot(data=df_md, x='Stage', y='MD',
               palette=['pink', 'green', 'orange', 'red'],
               inner='box')
sns.stripplot(data=df_md, x='Stage', y='MD',
              color='black', size=2, alpha=0.3)
plt.xlabel('')  # Remove x-axis label
plt.ylabel('MD (dB)')
plt.tight_layout()
plt.show()
plt.close()
# -------------------------------
# Pie Charts: Gender distribution across disease stages
# -------------------------------

# Define gender counts for each stage
gender_data = {
    'Early': {'Male': 534, 'Female': 664},
    'Moderate': {'Male': 270, 'Female': 385},
    'Advanced': {'Male': 240, 'Female': 295},
    'Severe': {'Male': 638, 'Female': 519}
}

# Draw 2x2 pie chart subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.flatten()
colors = ['deepskyblue', 'lightpink']

for i, (stage, counts) in enumerate(gender_data.items()):
    values = [counts['Male'], counts['Female']]
    labels = [f"Male: {counts['Male']}", f"Female: {counts['Female']}"]
    axes[i].pie(
        values,
        labels=labels,
        autopct=lambda p: f'{p:.2f}%' if p > 0 else '',
        colors=colors,
        startangle=90
    )
    axes[i].set_title(stage)

plt.tight_layout()
plt.show()
plt.close()
# -------------------------------
# KDE Plot: BCVA distribution across disease stages
# -------------------------------

# Simulate BCVA data (e.g., in logMAR units, lower is better)
bcva_data = {
    'Early': np.random.normal(loc=0.1, scale=0.05, size=100),
    'Moderate': np.random.normal(loc=0.3, scale=0.07, size=100),
    'Advanced': np.random.normal(loc=0.6, scale=0.08, size=100),
    'Severe': np.random.normal(loc=1.0, scale=0.1, size=100),
}

# Create a DataFrame
df_bcva = pd.DataFrame([(stage, value) for stage, values in bcva_data.items() for value in values],
                       columns=['Stage', 'BCVA'])

# Draw KDE density plot
plt.figure(figsize=(7, 5))
sns.kdeplot(data=df_bcva, x='BCVA', hue='Stage',
            fill=True, common_norm=False,
            palette=['skyblue', 'orange', 'green', 'red'])

# Reverse x-axis since lower BCVA values indicate better vision
plt.gca().invert_xaxis()
plt.xlabel("BCVA")
plt.ylabel("Density")

# Remove legend title only
plt.legend(title='')  # This removes the word "Stage" but keeps Early, Moderate, etc.

plt.tight_layout()
plt.show()
plt.close()