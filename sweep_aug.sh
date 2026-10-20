for aug in crop crop_erase; do
  AUG=$aug WANDB_TAGS=aug CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python main.py --cfg=configs/alexnet.yaml --opts TRAIN.LR 1e-3 DATA.BATCH_SIZE 256
done
