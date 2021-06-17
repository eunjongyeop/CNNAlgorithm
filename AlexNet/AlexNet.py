from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dropout, Dense
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from preprocessing import Preprocssing

batch_size = 128
momentum = 0.9
weight_decay = 0.05
lr = 0.001
epochs = 90

width = 227
height = 227
n_classes = 1000
# labels = {}

data_path = 'data'
load = False

# 이미지 로드
if not load:
    labels = {'berry':0, 'bird':1, 'dog':2, 'flower':3, 'other':4}
    prep = Preprocssing(width, height, labels=labels)
    paths = prep.load_paths(data_path)
    prep.load_imgs(paths)
    x, y, n_classes = prep.x, prep.y, prep.n_classes
    prep.save_data('data')
else:
    prep = Preprocssing(load=True, path=data_path)
    x, y = prep.x, prep.y
    width, height, n_classes, labels = prep.width, prep.height, prep.n_classes, prep.labels


# print([k for k, v in labels.items() if v == y[0]][0])

# 전처리
x = x.astype('float64')
y = to_categorical(y, num_classes=n_classes)

x /= 255.0

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.6, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42)

# 모델 생성
model = Sequential()
    # 1번째 layer
model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(width, height, 3), kernel_initializer=RandomNormal(mean=0, stddev=0.01, seed=42), bias_initializer='zeros', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))                                                                                                                         # 1번재 layer ↵
model.add(MaxPool2D((3, 3), strides=(2, 2)))
model.add(BatchNormalization())
    # 2번째 layer
model.add(Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer=RandomNormal(mean=0, stddev=0.01, seed=42), bias_initializer='zeros', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))
model.add(MaxPool2D((3, 3), strides=(2, 2)))
model.add(BatchNormalization())
    # 3번째 layer
model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=RandomNormal(mean=0, stddev=0.01, seed=42), bias_initializer='zeros', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))
    # 4번째 layer
model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=RandomNormal(mean=0, stddev=0.01, seed=42), bias_initializer='zeros', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))
    # 5번째 layer
model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=RandomNormal(mean=0, stddev=0.01, seed=42), bias_initializer='zeros', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))
model.add(MaxPool2D((3, 3), strides=(2, 2)))
model.add(Flatten())
    # 6번째 layer
model.add(Dense(4096, activation='relu', kernel_initializer='ones', bias_initializer='zeros', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))
model.add(Dropout(0.5))
    # 7번째 layer
model.add(Dense(4096, activation='relu', kernel_initializer='ones', bias_initializer='zeros', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))
model.add(Dropout(0.5))
    # 8번째 layer
model.add(Dense(n_classes, activation='softmax', kernel_initializer='ones', bias_initializer='zeros', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))
model.summary()

# 모델 학습
model.compile(optimizer=SGD(momentum=momentum, learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

reduce_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=lr*(0.1**3))
stop_cb = EarlyStopping(monitor='val_loss', patience=150)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[reduce_cb, stop_cb], validation_data=(x_val, y_val))

# 결과 보기
model.evaluate(x_test, y_test)