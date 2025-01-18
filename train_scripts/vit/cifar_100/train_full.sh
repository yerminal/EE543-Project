CUDA_VISIBLE_DEVICES=0,  python  -m torch.distributed.launch --nproc_per_node=1 --master_port=12346 \
	train.py ./cifar-100-python --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k \
	--dataset-download \
    --batch-size 128 --epochs 40 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-5 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output  output/vit_base_patch16_224_in21k/cifar_100/full \
	--amp  --pretrained  \