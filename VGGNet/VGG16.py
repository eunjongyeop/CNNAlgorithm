import os
import shutil
import random
from tqdm import tqdm

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD

train_path = 'data/train'
test_path = 'data/test'
test_rate = 0.2
img_width, img_height = 224, 224
n_class = 101

batch_size = 256
epoch = 10
momentum = 0.9
lr = 0.01

random.seed(42)

# 데이터 셋 로드
for dir_path in [train_path, test_path]:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

data = {}
for path, dir, file in os.walk('data/caltech101'):
    label = path.split('\\')[-1]

    if not file or label == 'Faces_easy':
        continue

    data[label] = file
label = list(data.keys())
n_label = len(label)

# print(data)
for key, item in tqdm(data.items()):
    p = (int)(len(item)*(1-test_rate))
    shuffle_item = item
    random.shuffle(shuffle_item)

    train, test = shuffle_item[:p], shuffle_item[p:]

    for dir_path in [train_path, test_path]:
        path = '{}/{}'.format(dir_path, key)

        if not os.path.exists(path):
            os.mkdir(path)
        if os.listdir(path):
            shutil.rmtree(path)
            os.mkdir(path)

    for filename in train:
        current_path = 'data/caltech101/{}/{}'.format(key, filename)
        move_path = 'data/train/{}/{}'.format(key, filename)
        shutil.copy(current_path, move_path)

    for filename in test:
        current_path = 'data/caltech101/{}/{}'.format(key, filename)
        move_path = 'data/test/{}/{}'.format(key, filename)
        shutil.copy(current_path, move_path)

# 전처리
train_data_generator = ImageDataGenerator(
    rescale= 1/.255,                # 1/255를 곱한다. - 스케일링
    rotation_range= 45,             # 좌우 반향 회전
    width_shift_range= 0.2,         # 범위 내 좌우이동
    height_shift_range= 0.2,        # 범위 내 상하이동
    horizontal_flip= True,          # 상하 반전
)

train_data_gen = train_data_generator.flow_from_directory(
    batch_size= batch_size,
    directory= train_path,
    target_size= (img_height, img_width),
    class_mode= 'categorical'
)

test_data_generator = ImageDataGenerator(
    rescale= 1/.255
)

test_data_gen = train_data_generator.flow_from_directory(
    batch_size= batch_size,
    directory= train_path,
    target_size= (img_height, img_width),
    class_mode= 'categorical'
)

# 모델 구축
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(img_height, img_width, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2), strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2), strides=2))
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(MaxPool2D((2, 2), strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2), 2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2), 2))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_class, activation='softmax'))

model.summary()

model.compile(optimizer=SGD(momentum=momentum, learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

# 학습 및 테스트
model.fit_generator(steps_per_epoch=10, generator=train_data_gen, validation_data=test_data_gen)