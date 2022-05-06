#!/bin/bash

# Make for COCOAPI
# cd cocoapi/PythonAPI
# make -j8
# cd ../..

post='SST-VOC-0.3'
printFreq=1000

mode='SST'
dataset='VOC2007'
prob=0.3

pretrainedModel='./data/checkpoint/resnet101.pth'
resumeModel='None'
evaluate='False'

epochs=20
startEpoch=0
stepEpoch=10

batchSize=8
lr=1e-5
momentum=0.9
weightDecay=5e-4

cropSize=448
scaleSize=512
workers=3

generateLabelEpoch=5

intraBCEWeight=1.0
intraBCEMargin=0.95
intraCooccurrenceWeight=10.0

interBCEWeight=1.0
interBCEMargin=0.95
interDistanceWeight=0.05
interExampleNumber=100

python SST.py \
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
    --generateLabelEpoch ${generateLabelEpoch} \
    --intraBCEMargin ${intraBCEMargin} \
    --intraBCEWeight ${intraBCEWeight} \
    --intraCooccurrenceWeight ${intraCooccurrenceWeight} \
    --interBCEWeight ${interBCEWeight} \
    --interBCEMargin ${interBCEMargin} \
    --interDistanceWeight ${interDistanceWeight} \
    --interExampleNumber ${interExampleNumber} \
