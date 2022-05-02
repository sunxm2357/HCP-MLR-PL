#!/bin/bash -l

#$ -N SST_0.6

#$ -m bea

#$ -M sunxm@bu.edu

# Set SCC project
#$ -P ivc-ml

# Request my job to run on Buy-in Compute group hardware my_project has access to
#$ -l buyin

# Request 4 CPUs
#$ -pe omp 3

# Request 2 GPU
#$ -l gpus=1

# Specify the minimum GPU compute capability
#$ -l gpu_c=7.5

#$ -l gpu_memory=48G

#$ -l h_rt=240:00:00

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load miniconda
module load cuda/11.1
module load gcc
conda activate CoOp
#conda install -c conda-forge opencv

export PROJECT_PATH=/projectnb/ivc-ml

export RESEARCH_PATH=/net/ivcfs4/mnt/data

#
cd $PROJECT_PATH/sunxm/code/HCP-MLR-PL/

nvidia-smi

post='SST-COCO-0.6'
printFreq=1000

mode='SST'
dataset='COCO2014'
prob=0.6

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
