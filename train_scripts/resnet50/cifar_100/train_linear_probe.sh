CUDA_VISIBLE_DEVICES=0,  python  -m torch.distributed.launch --nproc_per_node=1 --master_port=33518 \
	train.py ./cifar-100-python --dataset torch/cifar100 --num-classes 100 --model resnet50_in22k \
	--dataset-download \
    --batch-size 512 --epochs 40 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 5e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 5e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output  output/resnet50_in22k/cifar_100/linear_probe \
	--amp --tuning-mode linear_probe  --pretrained  \
	--log-wandb \
