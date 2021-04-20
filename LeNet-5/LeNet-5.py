from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import matplotlib.pyplot as plt

# 데이터 불려오기
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 전처리하기
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

    # 정규화
X_train = X_train / 255.0
X_test = X_test / 255.0

    # 원-핫 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 모델만들기
model = Sequential()
model.add(Conv2D(6, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D((2, 2), strides=(2, 2)))
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPool2D((2, 2), strides=(2, 2)))
model.add(Conv2D(120, (5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# 학습하기
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, epochs=30, validation_split=0.2)

# 평가하기
model.evaluate(X_test, y_test, batch_size=32)

# 결과 시각화하기
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.plot(history.history['loss'], 'b', label='train loss')
ax1.plot(history.history['val_loss'], 'r', label='val loss')
ax1.set_title('loss')
ax1.legend()
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')

ax2.plot(history.history['accuracy'], 'b', label='train accuracy')
ax2.plot(history.history['val_accuracy'], 'r', label='val accuracy')
ax2.set_title('accuracy')
ax2.legend()
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')

plt.show()