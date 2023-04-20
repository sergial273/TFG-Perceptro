import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD

# leer los datos desde el archivo
filename = 'datos.txt'
with open(filename, 'r') as f:
    lines = f.readlines()

# convertir cada línea en una entrada numérica de 64 x 7
inputs = []
for line in lines:
    # convertir la cadena de 448 bits en una lista de 64 elementos de 7 bits
    bits = [int(b) for b in line.strip().split(',')]
    mat = np.array(bits).reshape((64, 7))
    inputs.append(mat)
inputs = np.array(inputs)

# generar las salidas aleatorias
outputs = np.random.uniform(-1, 1, size=(len(lines), 1))

# definir la red neuronal
model = Sequential()
model.add(Flatten(input_shape=(64, 7)))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# compilar la red neuronal
model.compile(loss='mse', optimizer='adam')

# entrenar la red neuronal
epochs = 50
for i in range(epochs):
    history = model.fit(inputs, outputs, epochs=1, batch_size=32, verbose=1)
    print('Epoch', i+1, 'loss:', history.history['loss'][0])


# Evaluar la red neuronal
loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# Guardar el modelo en un archivo
model.save('red_neuronal.h5')

# Guardar los pesos entrenados en un archivo separado
model.save_weights('pesos.h5')


"""
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