from keras.models import Sequential
from keras.layers import Dense

# Crear la red neuronal
model = Sequential()
model.add(Dense(50, input_dim=267, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compilar la red neuronal
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


x_train, y_train = "bits de la posicio","sortida de la posicio"


x_test,y_test = "bits de la posicio per a comprobar funcionament","sortida de la posicio per a comprobar funcionament"


# Entrenar la red neuronal
model.fit(x_train, y_train, epochs=100, batch_size=32)

# Evaluar la red neuronal
loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))




"""
# Guardar el modelo en un archivo
model.save('red_neuronal.h5')

# Guardar los pesos entrenados en un archivo separado
model.save_weights('pesos.h5')


# Cargar el modelo desde el archivo
from keras.models import load_model
modelo = load_model('red_neuronal.h5')

# Cargar los pesos entrenados desde el archivo
modelo.load_weights('pesos.h5')
"""