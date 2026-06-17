# ==========================
# DATA PREPROCESSING TEMPLATE
# ==========================

# Import Libraries
import pandas as pd
import numpy as np

# Import Dataset
df = pd.read_csv("data.csv")

# Independent and Dependent Variables
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ==========================
# Taking Care of Missing Data
# ==========================
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

# ==========================
# Encoding Categorical Data
# ==========================
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_features = ['Sex', 'Embarked', 'Pclass']

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(df))

# ==========================
# Encoding Dependent Variable
# ==========================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df['Survived'])

# ==========================
# Splitting Dataset
# ==========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Feature Scaling
# ==========================
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# ==========================
# Output
# ==========================
print("X_train Shape :", X_train.shape)
print("X_test Shape  :", X_test.shape)
print("y_train Shape :", y_train.shape)
print("y_test Shape  :", y_test.shape)