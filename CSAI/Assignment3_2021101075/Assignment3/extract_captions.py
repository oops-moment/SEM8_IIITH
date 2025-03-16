import pandas as pd

from pycocotools.coco import COCO

stim_descriptions = pd.read_csv('nsd_stim_info_merged.csv',
                                index_col=0)

print("Important columns to see: 'cocoId', 'cocoSplit', 'nsdId'")
print("cocoId: ")

####
# E.g., I want to see the captions of training split image "train-0001_nsd-00007"
# train-0001 indicates whether this image belong to the training set or validation set
# nsd-00007 indicates nsdId

###
# Grab COCO subject id info that belong to nsdId 7

coco_id = stim_descriptions[stim_descriptions['nsdId'] == 7]['cocoId'].values[0]
coco_split = stim_descriptions[stim_descriptions['nsdId'] == 7]['cocoSplit'].values[0]

###
# Load COCO annotation data
coco_annotation_file = 'annotations_trainval2017/annotations/captions_{0}.json'.format(coco_split)
coco_data = COCO(coco_annotation_file)

coco_ann_ids = coco_data.getAnnIds(coco_id)
coco_annotations = coco_data.loadAnns(coco_ann_ids)
print(coco_annotations)
