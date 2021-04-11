python ./tensorboard-resnet.py -learning_rate 0.001 -batch_size 32 -epochs 5 &
python ./tensorboard-resnet.py -learning_rate 0.01 -batch_size 64 -epochs 5
wait
# tensorboard --logdir=/Users/henrygabrielyan/Desktop/python/ML/tensorboard/seq-resnet