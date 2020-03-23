#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np


# In[2]:


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


# In[3]:


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


# In[5]:


#### Test Cases ####
predicted_bboxes = np.array(
    [
        [0.0, 0.0, 200.0, 200.0], 
        [100.0, 100.0, 200.0, 200.0],
        [200.0, 200.0, 300.0, 300.0]
    ]
)
objects = {
            '101':{'w': 100, 'h': 100, 'y': 100, 'x': 100},
            '102':{'w': 100, 'h': 100, 'y': 200, 'x': 200},
            '103':{'w': 200, 'h': 200, 'y': 10, 'x': 10},
            '104':{'w': 200, 'h': 200, 'y': 0, 'x': 0},
          }

assert(matching(predicted_bboxes, objects) == ['104', '101', '102'])


# In[ ]:




