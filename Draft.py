var = 'YearBuilt'
data = pd.concat([bc['SiteEnergyUse(kBtu)'], bc[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SiteEnergyUse(kBtu)", data=data)
fig.axis(ymin=0, ymax=80000000)
plt.xticks(rotation=90)
plt.show()

var = 'YearBuilt'
data = pd.concat([bc['TotalGHGEmissions'], bc[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="TotalGHGEmissions", data=data)
fig.axis(ymin=0, ymax=3000)
plt.xticks(rotation=90)
plt.show()

import scipy.stats as stats
#bc['SiteEnergyUse(kBtu)'] = np.log(bc['SiteEnergyUse(kBtu)'])
res = stats.probplot(bc['SiteEnergyUse(kBtu)'], plot=plt)
plt.show()