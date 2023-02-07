model = dict(head=dict( num_classes=5, topk = (1, ) ))

_base_ = ['/data/home/scv9034/run/mmclassification/configs/_base_/models/resnet18.py', 
'/data/home/scv9034/run/mmclassification/configs/_base_/datasets/imagenet_bs32.py',
 '/data/home/scv9034/run/mmclassification/configs/_base_/default_runtime.py']

data = dict(samples_per_gpu = 32, workers_per_gpu=2, train = dict(data_prefix = 'data/flower/flower_dataset/train',
ann_file = '/data/home/scv9034/run/mmclassification/data/flower/flower_dataset/train.txt',
classes = '/data/home/scv9034/run/mmclassification/data/flower/flower_dataset/classes.txt'), 
val = dict(data_prefix = '/data/home/scv9034/run/mmclassification/data/flower/flower_dataset/val', 
ann_file = '/data/home/scv9034/run/mmclassification/data/flower/flower_dataset/val.txt',
 classes = '/data/home/scv9034/run/mmclassification/data/flower/flower_dataset/classes.txt'
),)

# 定义评估⽅法
evaluation = dict(metric_options={'topk': (1, )})

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=2)

load_from = '/data/home/scv9034/run/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'
