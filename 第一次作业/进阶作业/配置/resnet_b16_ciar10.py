model = dict(head=dict( num_classes=10, topk = (1, ) ))

_base_ = ['/data/home/scv9034/run/mmclassification/configs/_base_/models/resnet18_cifar.py', 
'/data/home/scv9034/run/mmclassification/configs/_base_/datasets/cifar10_bs16.py',
 '/data/home/scv9034/run/mmclassification/configs/_base_/default_runtime.py',
 '/data/home/scv9034/run/mmclassification/configs/_base_/schedules/cifar10_bs128.py']

data = dict(samples_per_gpu = 32, workers_per_gpu=2, train = dict(
data_prefix = '/data/home/scv9034/run/mmclassification/cifar-10-batches-py'), 
val = dict(data_prefix = '/data/home/scv9034/run/mmclassification/cifar-10-batches-py'),)

# 定义评估⽅法
evaluation = dict(metric_options={'topk': (1, )})
'''
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=2)
'''
load_from = '/data/home/scv9034/run/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'
