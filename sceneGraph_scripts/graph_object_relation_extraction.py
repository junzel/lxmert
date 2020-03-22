# ========================================= Note ======================================
# 1. validation set has object name 'pokemon', but training set doesn't


import json
import tqdm
import pdb

TRAIN_SCENE_GRAPHS = 'data/train_sceneGraphs.json'
VAL_SCENE_GRAPHS = 'data/val_sceneGraphs.json'

splits = ['train', 'val']

obj_list = []
obj_set = set()
relation_list = []
relation_set = set()
sub_obj_list = []
sub_obj_set = set()
obj_name_list = []
obj_name_set = set()

relation_dict = dict()

for split in splits:
    print(split)
    with open('data/{}_sceneGraphs.json'.format(split)) as f:
        data = json.load(f)
    for img_id, scene in tqdm.tqdm(data.items()):
        for obj in scene['objects']:
            # pdb.set_trace()
            obj_list.append(int(obj))
            obj_name_list.append(scene['objects'][obj]['name'])
            for relation in scene['objects'][obj]['relations']: # obj['relations'] is a list
                relation_list.append(relation['name'])
                if relation['name'] in relation_dict:
                    relation_dict[relation['name']] += 1
                else:
                    relation_dict[relation['name']] = 1
                sub_obj_list.append(int(relation['object']))
    obj_set = set(obj_list)
    relation_set = set(relation_list)
    sub_obj_set = set(sub_obj_list)
    obj_name_set = set(obj_name_list)
    print("Object list length:", len(obj_list))
    print("Number of unique objects:", len(obj_set))
    print("Relation list length:", len(relation_list))
    print("Number of unique relations:", len(relation_set))
    print("Object name list length:", len(obj_name_list))
    print("Number of unique object names:", len(obj_name_set))
   
    pdb.set_trace()
    
    # mapping = {'object_names': list(obj_name_set), 'relations': list(relation_set)}
    # with open('./{}_graph_mapping.json'.format(split), 'w') as f:
    #    json.dump(mapping, f)

