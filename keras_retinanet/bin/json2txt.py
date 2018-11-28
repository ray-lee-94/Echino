from pycocotools.coco import COCO
import json

dataset = COCO(annotation_file="/data/wen/Dataset/data_maker/COCO_maker/coco/annotations/instances_val2017.json")
# dataset.load_coco('/data/wen/Dataset/data_maker/COCO_maker/coco',"test")
# dataset.prepare()
# print(dataset.class_names)
result = json.load(open('val2017_bbox_results.json', 'r'))
with open('output.txt', 'w') as L:
    for instance in result:
        image_id = instance['image_id']
        image_name = dataset.loadImgs(image_id)
        image_name = image_name[0]['file_name']
        tp = dataset.cats[instance['category_id']]['name']
        bbox = list(map(int, instance['bbox']))
        bbox[2]+=bbox[0]
        bbox[3]+=bbox[1]
        bbox=list(map(str,bbox))
        score=str(instance['score'])
        L.writelines(" ".join([image_name,tp,bbox[0],bbox[1],bbox[2],bbox[3],score])+'\n')
