# Version 0.0.2 [![Build Status](https://travis-ci.org/izu-mi/py-tensor.svg?branch=dev)](https://travis-ci.org/izu-mi/py-tensor)
## Version 0.0.3 Changes: 
- Clean up code
- Docstrings
- CSV Crime Training Data
- Crime Testing Data
---
``` python
    """ This modules will make function calls from other packages """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# imports
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from utils import lstm
import time

#Load data and initilize variable to store data
X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', 50, True)
#Build model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True
))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False
))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1
))
model.add(Activation("linear"))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print("Compilation Time: ", time.time() - start)

#Train the model
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=1,
    validation_split=0.05
)

#Plot predictions!
predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
lstm.plot_results_multiple(predictions, y_test, 50)
```
