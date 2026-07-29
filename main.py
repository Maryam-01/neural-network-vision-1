import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import tensorflow as tf
from data import generate_blank, generate_vertical_line, generate_horizontal_line, generate_image_dataset


RANDOM_SEED = 1276
OUTPUT_EVERY_X_EPOCHS = 100

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Once again we have a function for printing our training progress. Theres a little more information printed in this one. 
print_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: 
    print(f"Epoch {epoch:3d} | Loss: {logs['loss']:.4f} | Accuracy: {logs['accuracy']:.4f} | "
          f"Val Loss: {logs['val_loss']:.4f} | Val Accuracy: {logs['val_accuracy']:.4f}")
    if epoch % OUTPUT_EVERY_X_EPOCHS == 0 else None
)
X, y = generate_image_dataset(100)

# Like during the decision tree notes, we randomly split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

# As before, build a multi-layer neural network
model = Sequential([
		Dense(16, activation='relu', input_shape=(64,)),  # Adds our first hidden layer.
		Dropout(0.1),  # Randomly disables 10% of neurons during each step, forcing it to not rely too much on any one neuron, preventing overfitting.
		Dense(8, activation='relu'),   # Second hidden layer
		Dense(3, activation='softmax')  # Output layer
        # We explain why ReLU and softmax are used later. 
])

# Compile the model like earlier. 
model.compile(optimizer=Adam(learning_rate=0.01),
							loss='sparse_categorical_crossentropy',
							metrics=['accuracy'])

# The training below has a few more arguments this time (batch_size: allows the training to go through the data in chunks, 32 at a time here, before updating the weights and biases to improve training speed and reduce memory usage). 
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0, callbacks=[print_callback])

# Once the model is trained we can do one final test of the model on the test data.
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Final Test Accuracy: {accuracy:.4%}")

print(X_test[89])


print("-----vertical------")
test_array = np.array([X_test[89]])
test_array_reshape = test_array.reshape(8,8)
print(test_array_reshape)

preds = model.predict(X_test[10:20])
predicted_label = np.argmax(preds, axis=1)

inputs = np.array(X_test[10:20])
outputs = np.array(y_test[10:20])
for image, label, preds in zip(inputs, outputs, predicted_label):
    train_reshaped = image.reshape(8,8)
    print("True label:", label, "Predicted_label:", preds)










