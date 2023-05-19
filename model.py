from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import TensorBoard
import matplotlib as plt
from preprocess import *

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(Dropout(0.5))  
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.5))  
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())  
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())  
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

model.save('model_weights.h5')

