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

##############################################################################
# Feature importance analysis with Random Forest
print("\nRandom Forest Feature Importance Analysis:")

model = RandomForestRegressor(n_estimators=100, random_state=666)
model.fit(X_train, y_train)

importances = model.feature_importances_

# Associer noms de colonnes + importance
feat_importances = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

print(feat_importances.head(10))

plt.figure(figsize=(10, 6))
plt.bar(feat_importances["feature"][:10], feat_importances["importance"][:10],
        color='skyblue', edgecolor='navy', alpha=0.7)

plt.title(f"Feature Importance - Random Forest ({predict})", fontsize=16, fontweight='bold')
plt.xlabel("Features", fontsize=12)
plt.ylabel("Importance", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


##############################################################################




# Get the fitted model from the last training for coefficient analysis
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), ['GFAByFloors', 'GFAByBuildings', 'PropertyGFAParking'])],
    remainder='passthrough'
)
pipeline = Pipeline([("preprocessing", preprocessor), ("model", LinearRegression())])
pipeline.fit(X_train, y_train)

# Extract coefficients (this is complex due to preprocessing)
# For visualization, we'll use the feature names from X
plt.figure(figsize=(8, 12))
# Note: coefficients order might be different due to ColumnTransformer
feature_names = X.columns
if hasattr(pipeline.named_steps['model'], 'coef_'):
    coef_values = pipeline.named_steps['model'].coef_
    plt.barh(range(len(feature_names)), coef_values)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title(f'Linear Regression Coefficients ({predict})')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


