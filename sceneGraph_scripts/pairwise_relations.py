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



