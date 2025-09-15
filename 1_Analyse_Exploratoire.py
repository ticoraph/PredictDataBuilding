import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

bc = pd.read_csv('seatle.csv')

# On regarde comment un batiment est défini dans ce jeu de données
#bc.head()
#print(bc.dtypes)
print(len(bc))

#print(bc['BuildingType'].value_counts(dropna=False))

bc = bc[~bc['BuildingType'].str.contains('Multifamily', na=False)]
print(len(bc))

########################################
######## DROPNA
########################################

bc.dropna(subset=['SiteEnergyUse(kBtu)'], inplace=True)
bc.dropna(subset=['TotalGHGEmissions'], inplace=True)
bc.dropna(subset=['Electricity(kWh)'], inplace=True)

bc['Electricity(kWh)'] = bc['Electricity(kWh)'].round()
bc['TotalGHGEmissions'] = bc['TotalGHGEmissions'].round()

########################################
######## OUTLIERS
########################################

def analyze_iqr(data, name="Dataset"):
    """Comprehensive IQR analysis of a dataset"""
    print(f"\n{name} Analysis:")
    print("-" * 30)

    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)  # Median
    q3 = np.percentile(data, 75)
    iqr = q3-q1

    print(f"Count: {len(data)}")
    print(f"Min: {min(data)}")
    print(f"Q1 (25%): {q1}")
    print(f"Q2 (50%, Median): {q2}")
    print(f"Q3 (75%): {q3}")
    print(f"Max: {max(data)}")
    print(f"IQR: {iqr}")
    print(f"IQR as % of range: {iqr / (max(data) - min(data)) * 100:.1f}%")

    # Outlier analysis
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_bound or x > upper_bound]

    print(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Number of outliers: {len(outliers)}")
    if outliers:
        print(f"Outlier values: {outliers}")
    return outliers

elec_outliers = analyze_iqr(bc['Electricity(kWh)'], "Electricity(kWh)")
co2_outliers = analyze_iqr(bc['TotalGHGEmissions'], "TotalGHGEmissions")

bc.drop(bc[bc["Electricity(kWh)"].isin(elec_outliers)].index, inplace=True)
bc.drop(bc[bc["TotalGHGEmissions"].isin(co2_outliers)].index, inplace=True)

########################################
########
########################################

print(bc['Electricity(kWh)'].sort_values(ascending=True).head(3))
print(bc['TotalGHGEmissions'].sort_values(ascending=True).head(3))
print(bc['Electricity(kWh)'].sort_values(ascending=False).head(3))
print(bc['TotalGHGEmissions'].sort_values(ascending=False).head(3))
print(bc['GHGEmissionsIntensity'].sort_values(ascending=False).head(3))
print(bc['NumberofFloors'].sort_values(ascending=False).head(3))

#bc = bc.drop(index=3274)
#bc = bc.drop(index=558)
#bc = bc.drop(index=1670)

# NumberofFloors
bc.loc[1359, 'NumberofFloors'] = 1

# TotalGHGEmissions outlier
bc = bc.drop(index=3206)

# GHGEmissionsIntensity outlier
bc = bc.drop(index=3373)
bc = bc.drop(index=3365)

bc.drop(bc[bc['SiteEnergyUse(kBtu)'] == 0].index, inplace=True)
bc.drop(bc[bc['TotalGHGEmissions'] == 0].index, inplace=True)

print(bc['LargestPropertyUseType'].value_counts(dropna=False))

#####################
# SiteEnergyUse(kBtu)
#####################

print('################### SiteEnergyUse(kBtu) #######################')
print(bc['SiteEnergyUse(kBtu)'].describe())
print("Skewness: %f" % bc['SiteEnergyUse(kBtu)'].skew())
print("Kurtosis: %f" % bc['SiteEnergyUse(kBtu)'].kurt())

#####################
# TotalGHGEmissions
#####################

print('################### TotalGHGEmissions #######################')
print(bc['TotalGHGEmissions'].describe())
print("Skewness: %f" % bc['TotalGHGEmissions'].skew())
print("Kurtosis: %f" % bc['TotalGHGEmissions'].kurt())

#####################
# FRIENDS CORRELATION
#####################

sns.scatterplot(x='SiteEnergyUse(kBtu)',y='TotalGHGEmissions',data=bc)
plt.show()


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.boxplot(bc['SiteEnergyUse(kBtu)'], vert=False)
plt.subplot(1, 2, 2)
plt.boxplot(bc['TotalGHGEmissions'], vert=False)
plt.tight_layout()
plt.show()


# On regarde le nombre de valeurs manquantes par colonne ainsi que leur type
bc.info()

sns.set_theme()
cols = ['SiteEnergyUse(kBtu)','Electricity(kBtu)','NaturalGas(kBtu)','TotalGHGEmissions', 'PropertyGFATotal']
sns.pairplot(bc[cols], size = 2)
plt.show()

#correlation matrix
numeric_cols = bc.select_dtypes(include=[np.number])
corrmat = numeric_cols.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,
           cmap='coolwarm',      # Better color scheme for correlations
           center=0,             # Center colormap at 0
           square=True,
           linewidths=0.5)       # Add grid lines
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

'''
occurrences = bc["YearBuilt"].value_counts().sort_index()
plt.bar(occurrences.index, occurrences.values)
plt.show()

ax = sns.scatterplot(x='TotalGHGEmissions',y='Electricity(kWh)', data=bc)
sns.regplot(x="TotalGHGEmissions", y="Electricity(kWh)",data=bc)
ax.set_title('Electricity vs CO2 REGRESSION')
plt.show()

#print(bc['PropertyGFABuilding(s)'].sort_values(ascending=False).head(10))

'''