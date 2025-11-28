.PHONY: build login push deploy terraform-apply all clean convert-models

IMAGE_NAME := animaldet
IMAGE_TAG ?= latest
AWS_REGION ?= us-east-1
TERRAFORM_DIR := infra/aws
ECR_REPOSITORY_URL ?= $(shell aws ecr describe-repositories --region $(AWS_REGION) --query 'repositories[?contains(repositoryName, `animaldet`)].repositoryUri' --output text 2>/dev/null | head -n1)

# Convert PyTorch models to ONNX
convert-models:
	@echo "==> Converting models to ONNX..."
	@python tools/convert_to_onnx.py --models all
	@echo "==> Conversion complete"

# Build Docker image
build:
	@echo "==> Building Docker image..."
	@docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "==> Build complete: $(IMAGE_NAME):$(IMAGE_TAG)"

# Login to ECR
login:
	@echo "==> Logging into ECR..."
	@aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(shell echo $(ECR_REPOSITORY_URL) | cut -d'/' -f1)
	@echo "==> Login successful"

# Push to ECR (requires ECR_REPOSITORY_URL)
push: build login
	@echo "==> Tagging and pushing image..."
	@docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(ECR_REPOSITORY_URL):$(IMAGE_TAG)
	@docker push $(ECR_REPOSITORY_URL):$(IMAGE_TAG)
	@echo "==> Image pushed: $(ECR_REPOSITORY_URL):$(IMAGE_TAG)"

# Apply Terraform configuration
terraform-apply:
	@echo "==> Applying Terraform..."
	@cd $(TERRAFORM_DIR) && terraform apply
	@echo "==> Terraform applied"

# Force ECS service update
ecs-update:
	@echo "==> Forcing ECS service update..."
	@aws ecs update-service --cluster animaldet-cluster --service animaldet-service --force-new-deployment --region $(AWS_REGION)
	@echo "==> ECS service updating..."

# Full deployment: build, push, and update infrastructure
deploy: push terraform-apply ecs-update
	@echo "==> Deployment complete!"

# Build and deploy everything
all: deploy

# Clean local Docker image
clean:
	@docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
