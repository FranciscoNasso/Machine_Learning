import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = 'melb_data.csv'
# Load the data
melbourne_data = pd.read_csv(melbourne_file_path)

# Print a summary of the data in Melbourne data
melbourne_data.describe()

melbourne_data.columns

# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price

melbourne_data_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_data_features]

X.describe()

X.head()

melbourne_model = DecisionTreeRegressor(random_state=1)

melbourne_model.fit(X, y)

# Making predictions for the first few rows of the training data to see how the predict function works
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))