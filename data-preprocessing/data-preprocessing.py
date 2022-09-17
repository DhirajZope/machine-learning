import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("Data.csv");

# Get all the data excluding last column beacause it is dependant data.
x = dataset.iloc[:, 0:-1].values

# Get first column because its a string datatype
y = dataset.iloc[:, -1].values

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[: , 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print("X: ", x)

# Process to transform dependant values into plot values so we can encode that value.

le = LabelEncoder()
y = le.fit_transform(y)

print("Y: ", y)



