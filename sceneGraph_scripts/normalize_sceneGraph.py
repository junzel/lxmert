# ===================================== Notes =================================
# 1. In scene graph json files, the highest key of the dicitonary is the image 
#    id (instead of question id).


import json
from pathlib import Path
import pdb

GQA_ROOT = './'

path = Path(GQA_ROOT + 'data')
split2name = {
    'train': 'train',
    'valid': 'val',
    # 'testdev': 'testdev',
    }
for split, name in split2name.items():
    new_data = []
    paths = [path / ("%s_sceneGraphs.json" % name)]
    print(split, paths)

    for tmp_path in paths:
        with tmp_path.open() as f:
            data = json.load(f)
            for key, datum in data.items():
                width = datum['width']
                height = datum['height']
                for obj_id, obj in datum['objects'].items():
                    # if obj['x'] + obj['w'] > width or obj['y'] + obj['h'] > height:
                    #     print("should use minus")
                    #     pdb.set_trace()
                    # elif obj['x'] - obj['w'] < 0 or obj['y'] - obj['h'] < 0:
                        # print("should use plus")
                        # pdb.set_trace()
                    obj['x'] = obj['x'] / width
                    obj['y'] = obj['y'] / height
                    # not sure if image's top left corner is (0, 0) or bottom left corner is, it determins '-' or '+'
                    obj['box'] = [obj['x'], obj['y'], obj['x']+obj['w']/width, obj['y']+obj['h']/height] # [top-left-x, top-left-y, right-bottom-x, right-bottom-y]
    # pdb.set_trace()
    json.dump(data, open("./normalized_%s_sceneGraphs.json" % split, 'w'),
                          indent=4, sort_keys=True)
