hello:
	echo "hello world"

build:
	docker build \
	-t traffik:gpu \
	--build-arg WANDB_API_KEY=${WANDB_API_KEY} \
	--build-arg WANDB_PROJECT=${WANDB_PROJECT} \
	--build-arg WANDB_USERNAME=${WANDB_USERNAME} .
