NAME="UNIT_200epoch_again"
DATASET="pokemon-sprites"
CHANNELS=3


nohup python ./unit/unit.py --experiment_name $NAME --batch_size 2 \
 --dataset_name $DATASET --channels $CHANNELS --checkpoint_interval 2 &> log.txt
