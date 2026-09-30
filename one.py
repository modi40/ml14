import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

try:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
except Exception:
    np.random.seed(42)
    X_train = np.random.randint(0, 255, (200, 28, 28), dtype=np.uint8)
    y_train = np.random.randint(0, 10, 200)
    X_test = np.random.randint(0, 255, (50, 28, 28), dtype=np.uint8)
    y_test = np.random.randint(0, 10, 50)

X_train_sub = X_train[:300].astype('float32') / 255.0
y_train_sub = y_train[:300]
X_test_sub = X_test[:50].astype('float32') / 255.0
y_test_sub = y_test[:50]

X_train_sub = np.expand_dims(X_train_sub, -1)
X_test_sub = np.expand_dims(X_test_sub, -1)

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(16, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_sub, y_train_sub, epochs=1, batch_size=64, verbose=0)

sample_image = X_test_sub[0:1]
actual_digit = y_test_sub[0]

predicted_probs = model.predict(sample_image, verbose=0)
predicted_digit = np.argmax(predicted_probs)

print(f"Actual Handwritten Digit in Image: {actual_digit}")
print(f"Predicted Digit by CNN Model: {predicted_digit}")
