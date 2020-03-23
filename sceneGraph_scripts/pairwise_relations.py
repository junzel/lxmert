import torch
import numpy as np
import json
import pdb
import argparse
import time
import pickle
import csv
import sys, os
import base64

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
                      "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
# scene graph json file paths
TRAIN_SCENE_GRAPHS = 'data/normalized_train_sceneGraphs.json'
VAL_SCENE_GRAPHS = 'data/normalized_valid_sceneGraphs.json'
GRAPH_MAPPING = 'data/graph_mapping.json'

def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

def bb_iou(boxA, boxB):
    """Calculate the Intersection of Union of boxes from Faster RCNN and scene graph"""
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def matching(predicted_bboxes, objects):
    '''
    ###input type:###
    
    ###对于每个img_id都有predicted_bboxes, objects###
    
    1. predicted_bboxes: numpy array, 36x4, 每张图faster rcnn predict出来的36个bbox
    
    2. objects: dict, scenegraph 中img_id包含所有的object
        e.g.:
            {'2716708'(object_id): 
                {
                    'name': 'nose',
                    'attributes': [],
                    'relations': [{'object': '2449847', 'name': 'of'},
                               {'object': '2493560', 'name': 'to the right of'},
                               {'object': '2370325', 'name': 'to the left of'}],

                    'w': 7,
                    'h': 9,
                    'y': 54,
                    'x': 370},
            }
            
    '''
    ious = np.zeros((predicted_bboxes.shape[0], len(objects)))
    object_ids = [""]*predicted_bboxes.shape[0]
    for i in range(predicted_bboxes.shape[0]):
        for j, obj_id in enumerate(objects.keys()):
            x = float(objects[obj_id]['x'])
            y = float(objects[obj_id]['y'])
            w = float(objects[obj_id]['w'])
            h = float(objects[obj_id]['h'])
            obj_bbox = [x, y, x + w, y + h]
            ious[i, j] = bb_iou(predicted_bboxes[i], obj_bbox)
        
    objects_keys = list(objects.keys())        
    for i in range(predicted_bboxes.shape[0]):
        max_iou = np.unravel_index(np.argmax(ious), ious.shape)

        obj_id = objects_keys[max_iou[1]]
        objects_keys.pop(max_iou[1])

        object_ids[max_iou[0]] = obj_id
        ious = np.delete(ious, max_iou[1], 1)
    
    '''        
    ###return type:###   
    
    object_ids: list, length = 36, 返回每个predicted_bboxes对应的objects中的object_id (type: string)
            
    '''
    
    return object_ids

def output_normalized_bboxes(img_data):
    """Normalize bboxes to [0, 1]"""
    # Normalize the boxes (to 0 ~ 1)
    boxes = img_data['boxes'].copy()
    img_h, img_w = img_data['img_h'], img_data['img_w']
    boxes = boxes.copy()
    boxes[:, (0, 2)] /= img_w
    boxes[:, (1, 3)] /= img_h

    return boxes

def relation_tensor(object_ids, scene_graph_objects, relation_mapping):
    """
    object_ids (list of dict): each bbox's matching object id
    scene_graph (dict): scene graph loaded from json file

    """
    tensor = torch.zeros([36, 36, 311], dtype=torch.int32)
    tesnor[:, :, -1] = 1
    for idx, obj in enumerate(object_ids):
        if obj not in scene_graph_objects or obj == None:
            pass
        else:
            object_relations = scene_graph_objects[obj]['relations']
            if len(object_relations) < 1:
                pass
            else:
                for relation in object_relations:
                    related_obj = relation['object']
                    try:
                        related_idx = object_ids.index(related_obj)
                    except:
                        continue
                    tensor[idx, related_idx, relation_mapping.index(relation['name'])]
    return tensor


def create_tensor():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=['train', 'valid'], default='train', type=str)
    args = parser.parse_args()

    # Load output of Faster R-CNN
    if args.split == 'valid':
        path = "../data/vg_gqa_imgfeat/gqa_testdev_obj36.tsv"
    else:
        path = "../data/vg_gqa_imgfeat/vg_gqa_obj36.tsv"
    img_data = load_obj_tsv(path)
    imgid2img = {}
    for img_datum in img_data:
        imgid2img[img_datum['img_id']] = img_datum

    output = dict()
    # Loading scene graphs to imgid2scenegraph
    if args.split == 'valid':
        with open(VAL_SCENE_GRAPHS) as f:
            imgid2scenegraph = json.load(f)
    else:
        with open(TRAIN_SCENE_GRAPHS) as f:
            imgid2scenegraph = json.load(f)
    with open(GRAPH_MAPPING) as f:
        graph_mapping = json.load(f)
    
    relation_mapping = graph_mapping['relations']

    for img_id, scene_graph in imgid2scenegraph.items():
        scene_graph_objects = scene_graph['objects']
        bboxes = output_normalized_bboxes(imgid2img[img_id])
        object_ids = matching(bboxes, scene_graph_objects)
        tensor = relation_tensor(object_ids[img_id], scene_graph_objects, relation_mapping)
        
        output[img_id] = tensor

    pdb.set_trace()
    with open('pairwise_relation_tensor.pkl') as f:
        pickle.dump(output, f)

    return output

if __name__ == "__main__":
    print("Hello")
    _ = create_tensor()



