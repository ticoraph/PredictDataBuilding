import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

building_consumption = pd.read_csv('seatle.csv')


# On regarde comment un batiment est défini dans ce jeu de données
#building_consumption.head()
print(len(building_consumption))

print(building_consumption['BuildingType'].value_counts(dropna=False))

building_consumption = building_consumption[~building_consumption["BuildingType"].str.contains("Multifamily", na=False)]

print(len(building_consumption))


# On regarde le nombre de valeurs manquantes par colonne ainsi que leur type
#building_consumption.info()