'''生成参照框train_box test_box'''

import base64
import csv
import numpy as np

csv.field_size_limit(500 * 1024 * 1024)

feature = {}
FIELDNAMES = ['image_id', 'image_name', 'image_w', 'image_h', 'num_boxes', 'boxes', 'pred_scores']

if __name__ == '__main__':
    with open('train_objects.csv', "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        sum = 0
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes']:  # , 'pred_scores'
                data = item[field]
                # 这里是将csv中存储的数据转换为np.array
                buf = base64.b64decode(data[1:])
                temp = np.frombuffer(buf, dtype=np.float32)
                item[field] = temp.reshape((item['num_boxes'], -1))
            a = item['boxes']
            b = np.argsort(-a[:, 5])
            a = a[b, :]
            d = a[:5, :]
            feature[item['image_name']] = d
            sum += 1
            print(sum)

    np.save("train_box.npy", feature)

    with open('test_objects.csv', "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        sum = 0
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes']:  # , 'pred_scores'
                data = item[field]
                # 这里是将csv中存储的数据转换为np.array
                buf = base64.b64decode(data[1:])
                temp = np.frombuffer(buf, dtype=np.float32)
                item[field] = temp.reshape((item['num_boxes'], -1))
            a = item['boxes']
            b = np.argsort(-a[:, 5])
            a = a[b, :]
            d = a[:5, :]
            feature[item['image_name']] = d
            sum += 1
            print(sum)

    np.save("test_box.npy", feature)
