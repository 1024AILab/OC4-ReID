# The code is builded with DistributedDataParallel. 
# Reprodecing the results in the paper should train the model on 2 GPUs.
# You can also train this model on single GPU and double config.DATA.TRAIN_BATCH in configs.
# For LTCC dataset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset occ_ltcc --cfg configs/res50_cels_cal.yaml --gpu 0,1 #
# For PRCC dataset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset occ_prcc --cfg configs/res50_cels_cal.yaml --gpu 0,1 #
