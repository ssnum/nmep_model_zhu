for bs in 64 100 512; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python main.py --cfg=configs/alexnet.yaml --opts TRAIN.LR 1e-3 DATA.BATCH_SIZE $bs
done
