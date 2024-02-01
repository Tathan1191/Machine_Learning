import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Load CSV data frame pandas

data = pd.read_csv('ejemplo.csv')

# Dividir dataframe en cacteristicas y etiquetas
x = data [['peso', 'sexo', 'altura']]
y = data ['talla']


# Crear una lista de modelos
models = [
    ('decision_tree',   DecisionTreeRegressor()),
    ('linear_regression', LinearRegression()),
    ('k_neighbors', KNeighborsRegressor(n_neighbors=5))
]


# Modelo ensable modelos anteriores
model = VotingRegressor (models)

# Entreno modelo con los datos

model.fit(x,y)

