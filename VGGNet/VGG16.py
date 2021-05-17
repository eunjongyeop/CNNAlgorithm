import os, cv2
import numpy as np

# 데이터 셋 로드
data = {}
for path, dir, file in os.walk('data'):
    label = path.split('\\')[-1]

    if not file or label == 'Faces_easy':
        continue

    data[label] = file
label = list(data.keys())
n_label = len(label)

for key, item in data.items():
    for filename in item:
        print(f'data\\catech101\\{key}\\{filename}')

# 전처리
# 모델 구축
# 학습 및 테스트