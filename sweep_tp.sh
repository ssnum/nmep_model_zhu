for bs in 1 2 4 8 16; do
  echo "===== batch size $bs ====="
  WANDB_TAGS=throughput CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python main.py --cfg=configs/alexnet.yaml --opts TRAIN.LR 1e-3 DATA.BATCH_SIZE $bs TRAIN.EPOCHS 1 || break
done
for bs in 32 64 128 256 512 1024 2048 4096 8192 16384 32768; do
  echo "===== batch size $bs ====="
  WANDB_TAGS=throughput CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python main.py --cfg=configs/alexnet.yaml --opts TRAIN.LR 1e-3 DATA.BATCH_SIZE $bs TRAIN.EPOCHS 3 || break
done
