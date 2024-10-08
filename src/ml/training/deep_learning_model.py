from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from src.util.feature_names import get_features_name

def build_model(units_list=[64], activation='relu', optimizer='adam', learning_rate=0.001):
    input_dim = len(get_features_name())  # Dynamically get input_dim
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # Use Input layer to specify input shape
    
    # Add multiple Dense layers
    for units in units_list:
        model.add(Dense(units=units, activation=activation))  # Add Dense layers

    model.add(Dense(1))  # Output layer for regression
    if optimizer == 'adam':
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    else:
        # Use SGD or other optimizers if needed
        model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Create the KerasRegressor and pass the input_dim dynamically during fit
keras_reg = KerasRegressor(model=build_model, verbose=0)