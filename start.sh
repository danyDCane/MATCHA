mpirun -np 8 \
-bind-to none \
-map-by slot \
-mca pml ob1 -mca btl ^openib \
python train_mpi.py \
--lr 0.05 \
--bs 64 \
--epoch 200 \
--budget 1.0 \
--momentum 0.9 \
--warmup \
-n Vanilla_DecenSGD \
--model res \
-p \
--description Vanilla_DecenSGD_experiment \
--graphid 0 \
--dataset cifar10 \
--datasetRoot ../AdaDSGD/data/ \
--savePath ./exp_result_ \
--randomSeed 1234