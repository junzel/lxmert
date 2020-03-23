import torch
import numpy as np
import json
import pdb
import argparse

import pickle


# scene graph json file paths
TRAIN_SCENE_GRAPHS = 'data/gqa/normalized_train_sceneGraphs.json'
VAL_SCENE_GRAPHS = 'data/gqa/normalized_valid_sceneGraphs.json'
GRAPH_MAPPING = 'data/gqa/graph_mapping.json'

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


def create_tensor()
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choice=['train', 'valid'], default='train', type=str)
    args = parser.parse_args()

    output = dict()
    # Loading scene graphs to imgid2scenegraph
    if args['split'] == 'valid':
        with open(VAL_SCENE_GRAPHS) as f:
            imgid2scenegraph = json.load(f)
    else:
        with open(TRAIN_SCENE_GRAPHS) as f:
            imgid2scenegraph = json.load(f)
    with open(GRAPH_MAPPING) as f:
        graph_mapping = json.load(f)
    
    relation_mapping = graph_mapping['relations']
    scene_graph_objects = scene_graph['objects']

    for img_id, scene_graph in imid2scenegraph.items():
        tensor = relation_tensor(object_ids[img_id], scene_graph_objects, relation_mapping)
        pdb.set_trace()
        output[img_id] = tensor

    with open('pairwise_relation_tensor.pkl') as f:
        pickle.dump(output, f)

    return output

if __name__ == '__main__':
    _ = create_tensor()



