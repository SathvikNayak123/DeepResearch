#!/usr/bin/env bash
# Build the current image, push it to the ECR repo terraform created, and
# force ECS to roll a new deployment picking it up. Must be run after
# `terraform apply` (reads outputs from infra/).
set -euo pipefail

cd "$(dirname "$0")/.."

REPO_URL=$(terraform -chdir=infra output -raw ecr_repository_url)
CLUSTER=$(terraform -chdir=infra output -raw ecs_cluster_name)
SERVICE=$(terraform -chdir=infra output -raw ecs_service_name)
REGION=$(terraform -chdir=infra output -raw ecr_repository_url | cut -d. -f4)
TAG="${1:-latest}"

echo "Building image..."
docker build -t "${REPO_URL}:${TAG}" .

echo "Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO_URL%/*}"

echo "Pushing ${REPO_URL}:${TAG}..."
docker push "${REPO_URL}:${TAG}"

echo "Forcing new ECS deployment on ${CLUSTER}/${SERVICE}..."
aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --force-new-deployment --region "$REGION" >/dev/null

echo "Waiting for the service to stabilize..."
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION"

echo "Deployed. Live URL: $(terraform -chdir=infra output -raw alb_url)"
