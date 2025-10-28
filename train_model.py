import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
import joblib
import os

print("Starting model training...")

# Create models folder if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load data
df = pd.read_csv('fuel_prices_UK.csv')

# Convert date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Feature engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Rolling features
df['RollingMean_3'] = df.groupby('City').apply(
    lambda x: x['Price_GBP'].rolling(window=3, min_periods=1).mean()
).reset_index(level=0, drop=True)

df['RollingStd_3'] = df.groupby('City').apply(
    lambda x: x['Price_GBP'].rolling(window=3, min_periods=1).std()
).reset_index(level=0, drop=True)

df['RollingStd_3'] = df['RollingStd_3'].fillna(0)

# Label encoding
le_city = LabelEncoder()
le_county = LabelEncoder()
le_fuel = LabelEncoder()

df['City_encoded'] = le_city.fit_transform(df['City'])
df['County_encoded'] = le_county.fit_transform(df['County'])
df['Fuel_Type_encoded'] = le_fuel.fit_transform(df['Fuel_Type'])

# Save label encoders
joblib.dump(le_city, 'models/le_city.pkl')
joblib.dump(le_county, 'models/le_county.pkl')
joblib.dump(le_fuel, 'models/le_fuel.pkl')

# Create price category
median_price = df['Price_GBP'].median()
df['Price_Category'] = (df['Price_GBP'] > median_price).astype(int)

# Save median price
joblib.dump(median_price, 'models/median_price.pkl')

# Features
features = ['City_encoded', 'County_encoded', 'Fuel_Type_encoded', 
            'Month', 'Day', 'DayOfWeek', 'RollingMean_3', 'RollingStd_3']

# Prepare data for regression
X_reg = df[features]
y_reg = df['Price_GBP']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)

# Train Linear Regression
print("Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_reg_scaled, y_train_reg)
joblib.dump(lr_model, 'models/linear_regression.pkl')
joblib.dump(scaler_reg, 'models/scaler_regression.pkl')

# Prepare data for classification
X_clf = df[features]
y_clf = df['Price_Category']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)

# Train Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_clf_scaled, y_train_clf)
joblib.dump(rf_model, 'models/random_forest.pkl')

# Train AdaBoost
print("Training AdaBoost...")
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train_clf_scaled, y_train_clf)
joblib.dump(ada_model, 'models/adaboost.pkl')

# Train Gradient Boosting
print("Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_clf_scaled, y_train_clf)
joblib.dump(gb_model, 'models/gradient_boosting.pkl')

joblib.dump(scaler_clf, 'models/scaler_classification.pkl')

# Train K-Means Clustering
print("Training K-Means...")
cluster_features = ['Price_GBP', 'City_encoded', 'County_encoded', 'Fuel_Type_encoded', 'Month']
X_cluster = df[cluster_features]

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_cluster_scaled)
joblib.dump(kmeans, 'models/kmeans.pkl')
joblib.dump(scaler_cluster, 'models/scaler_clustering.pkl')

# Save unique values for dropdowns
joblib.dump(df['City'].unique().tolist(), 'models/cities.pkl')
joblib.dump(df['County'].unique().tolist(), 'models/counties.pkl')

print("\n✓ All models trained and saved successfully!")
print("Models saved in 'models/' folder")
