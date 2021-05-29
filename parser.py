import json
import csv
import os
from PIL import Image

list_ann = []
with open('train.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        box_number = 0
        numbers = row[2].split()
        for box_number_i in range(round(len(numbers) / 4)-1):
            bbox = "[{}, {}, {}, {}]".format(numbers[box_number], numbers[box_number+1], numbers[box_number+2], numbers[box_number+3])
            peace = {'area': 0,
                'bbox':bbox,
                    'category_id': int(row[1]),
                    'id': box_number_i,
                        'image_id': int(row[0]),
                        'iscrowd': 0,
                            'segmentation': []}
            list_ann.append(peace)
            print(peace)
            box_number += 4

list_images = []
#id_im = 0
for subdir, dirs, files in os.walk('./train_images'):
    for file in files:
        im = Image.open("./train_images/{}".format(file))
        (width, height) = im.size
        imgs = {'coco_url': '',
            'date_captured': '2020/12/9',
                'file_name': file,
                    'flickr_url': '',
                        'height': height,
                            'id': int(file),
                                'license': 1,
                                    'width': width}
    #print(imgs)
        #id_im += 1
        list_images.append(imgs)

annotations = json.dumps({'categories': [{"id": 1, "name": "1", "supercategory": "None"}, {"id": 2, "name": "2", "supercategory": "None"}, {"id": 3, "name": "3", "supercategory": "None"}, {"id": 4, "name": "4", "supercategory": "None"}], 'images': list_images, 'annotations': list_ann, 'info': {'contributor': '',
                         'data_created': '2020-12-09',
                         'description': '',
                         'url': '',
                         'version': '',
                         'year': 2020}, 'licenses': [{'id': 1, 'name': None, 'url': None}]})
#print(annotations)

file = open("instances_trainsteel.json", "w")
file.write(annotations)
file.close()

print("NEWVERSION\n")
