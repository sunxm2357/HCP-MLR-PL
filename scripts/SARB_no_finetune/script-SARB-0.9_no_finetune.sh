#!/bin/bash

post='SARB-COCO-0.9'
printFreq=1000

mode='SARB'
dataset='COCO2014'
prob=0.9

pretrainedModel='./data/checkpoint/resnet101.pth'
resumeModel='None'
evaluate='False'

epochs=20
startEpoch=0
stepEpoch=10
workers=8

batchSize=32
lr=1e-5
momentum=0.9
weightDecay=5e-4

cropSize=448
scaleSize=512
workers=8

mixupEpoch=5
contrastiveLossWeight=0.05

prototypeNum=10
recomputePrototypeInterval=5

isAlphaLearnable='True'
isBetaLearnable='True'

python SARB.py \
    --post ${post} \
    --printFreq ${printFreq} \
    --mode ${mode} \
    --dataset ${dataset} \
    --prob ${prob} \
    --pretrainedModel ${pretrainedModel} \
    --resumeModel ${resumeModel} \
    --evaluate ${evaluate} \
    --epochs ${epochs} \
    --startEpoch ${startEpoch} \
    --stepEpoch ${stepEpoch} \
    --batchSize ${batchSize} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weightDecay ${weightDecay} \
    --cropSize ${cropSize} \
    --scaleSize ${scaleSize} \
    --workers ${workers} \
    --mixupEpoch ${mixupEpoch} \
    --contrastiveLossWeight ${contrastiveLossWeight} \
    --prototypeNum ${prototypeNum} \
    --recomputePrototypeInterval ${recomputePrototypeInterval} \
    --isAlphaLearnable ${isAlphaLearnable} \
    --isBetaLearnable ${isBetaLearnable} \
