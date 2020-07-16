from collections import defaultdict

vg150_hierarchy = [['__background__'],
['boot', 'cap', 'coat', 'glass', 'glove', 'hat', 'helmet', 'jacket', 'jean', 'pant', 'shirt', 'shoe', 'short', 'sneaker', 'sock', 'tie'],
['arm', 'ear', 'eye', 'finger', 'head', 'leg', 'mouth', 'neck', 'nose', 'paw', 'tail', 'trunk', 'hair', 'hand', ],
[ 'guy',  'man', 'player', 'skier', 'person', 'lady', 'woman'],
['kid', 'boy', 'child', 'girl', ],
['animal', 'cow', 'elephant', 'giraffe', 'sheep', 'zebra', 'bear', 'cat', 'dog',  'horse',],
['bird',  'kite', 'roof'],
['boat', 'car', 'train' , 'truck', 'bus', 'vehicle'],
['airplane', 'engine', 'face',  'handle', 'plane', 'tire', 'wheel', 'windshield', 'wing'],
['leaf', 'letter', 'logo', 'number'],
['lamp', 'book', 'bottle', 'cup', 'paper', 'phone', 'pizza', 'towel', 'vegetable', 'fruit', 'orange', 'bag', 'banana', 'food', ],
['cabinet', 'counter', 'curtain', 'desk', 'drawer', 'toilet', 'post', 'shelf', 'table',],
['bench', 'chair' , 'railing', 'rock', 'sidewalk', 'street', 'wire'],
['basket', 'bowl', 'box', 'flower', 'pot', 'vase'],
['bed', 'plant', 'sink', 'snow', 'stand', 'track',  'wave', 'pole', 'ski', 'surfboard', ],
['racket'],
['room'],
['fence', 'hill', 'house', 'men', 'mountain', 'people', 'tree'],
['board', 'clock', 'laptop', 'pillow', 'plate', 'sign', 'tile', ],
['beach', 'bike', 'branch', 'building',  'door', 'light', 'motorcycle', 'screen', 'seat', 'skateboard', 'tower',  'window'],
[ 'flag', 'fork', 'umbrella']]

def clusterIds(origin_list, hierarchy):
    cluster_ids = defaultdict(list)
    for cid, c in enumerate(hierarchy):
        for cc in c:
            cluster_ids[cid].append(origin_list.index(cc))
    originids_to_clusterids = {}
    for k, v in cluster_ids.items():
        for vv in v:
            originids_to_clusterids[vv] = k
    return originids_to_clusterids

