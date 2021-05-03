import numpy as np
import cv2
import os
from json import dump, load


def concat_path(*paths, dash='/'):
    if len(paths) == 1:
        return paths
    return dash.join(paths)

def print_process(size, complete, content):
    """
    :param size: int
        현재 진행 중인 프로세스의 길이
    :param complete: int
        완료된 프로세스
        진행 상황은 complete / size %로 정하며 int값을 가진다.
    :param content: str
        파일 경로 등 추가하고 싶은 문자열
    :return: None
        진행상황을 출력한다.
    """

    process_len = 50

    percent = round(complete / size * 100)
    process = '=' * (percent // (100 // process_len))
    if len(process) < process_len:
        process += '>'
    process += '.' * (process_len - len(process))

    print('\r{}/{} [{}] {}% {}'.format(complete, size, process, percent, content), end='')


class Preprocssing:
    def __init__(self, width=None, height=None, labels={}, load=False, path=None):
        self.save_file_names = {'dir' : 'preprocessed', 'x' : 'x_data.npy', 'y' : 'y_data.npy', 'meta' : 'meta.json'}

        if not load:
            self.width = width
            self.height = height
            self.n_classes = len(labels)

            self.labels = labels

            self.x, self.y = np.empty((0, width, height, 3)), np.empty((0))
        else:
            self.x, self.y, meta = self.load_data(path)

            self.width = meta['width']
            self.height = meta['height']
            self.n_classes = meta['n_classes']
            self.labels = meta['labels']

    def load_paths(self, path):
        paths = []


        for root, dirs, files in os.walk(path):
            for file in files:
                if file.split('.')[-1] in ['jpg', 'JPG', 'png', 'PNG']:
                    paths.append(concat_path(root, file, dash='\\'))
        return paths

        return load_paths_res(path)

    def load_imgs(self, paths):
        def read_img(path):
            img = cv2.imread(path)
            img = cv2.resize(img, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA).reshape(
                (1, self.width, self.height, 3))
            return img

        size = len(paths)
        complete = 0
        for path in paths:
            complete += 1
            print_process(size, complete, path)

            self.x = np.append(self.x, read_img(path))
            self.y = np.append(self.y, np.array([path.split('\\')[-2]]), axis=0)
        print()
        self.labels = self.get_label(self.y)

        self.x = self.x.reshape((-1, self.width, self.height, 3))
        self.y = np.array([self.labels[l] for l in self.y])

    def get_label(self, y):
        labels = dict()
        for i, label in enumerate(set(y)):
            labels[label] = i

        return labels

    def save_data(self, path):
        if self.save_file_names['dir'] not in os.listdir(path):
            os.mkdir(concat_path(path, self.save_file_names['dir']))

        np.save(concat_path(path, self.save_file_names['dir'], self.save_file_names['x']), self.x)
        np.save(concat_path(path, self.save_file_names['dir'], self.save_file_names['y']), self.y)

        meta = dict()
        meta['width'] = self.width
        meta['height'] = self.height
        meta['n_classes'] = self.n_classes
        meta['labels'] = self.labels
        with open(concat_path(path, self.save_file_names['dir'], self.save_file_names['meta']), 'w') as json_file:
            dump(meta, json_file)

    def load_data(self, path):
        try:
            x = np.load(concat_path(path, self.save_file_names['dir'], self.save_file_names['x']))
            y = np.load(concat_path(path, self.save_file_names['dir'], self.save_file_names['y']))
            with open(concat_path(path, self.save_file_names['dir'], self.save_file_names['meta']), 'r') as json_file:
                meta = load(json_file)
        except Exception as e:
            raise e

        return x, y, meta