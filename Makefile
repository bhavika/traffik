.PHONY: hello cpu gpu venv lint
SHELL := /bin/bash

hello:
	echo "hello world"

cpu:
	docker build \
	-t traffik:cpu \
	--build-arg WANDB_API_KEY=${WANDB_API_KEY} \
	--build-arg WANDB_PROJECT=${WANDB_PROJECT} \
	--build-arg WANDB_USERNAME=${WANDB_USERNAME} .

gpu:
	docker build -f Dockerfile.gpu \
	-t traffik:gpu \
	--build-arg WANDB_API_KEY=${WANDB_API_KEY} \
  	--build-arg WANDB_PROJECT=${WANDB_PROJECT} \
	--build-arg WANDB_USERNAME=${WANDB_USERNAME} .

venv:
	source venv/bin/activate

lint:
	black traffik/
