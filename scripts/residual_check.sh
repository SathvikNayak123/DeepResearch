#!/usr/bin/env bash
# Post-destroy sweep: `terraform destroy` succeeding is not proof nothing was
# left behind (a resource outside terraform's state, or a destroy that
# silently no-ops on one resource, wouldn't show up in its exit code).
# Queries AWS directly for anything still tagged project=deepresearch (or
# matching the naming prefix for the handful of resource types the Resource
# Groups Tagging API doesn't cover) and exits non-zero if it finds any.
set -euo pipefail

PROJECT="${PROJECT:-deepresearch}"
REGION="${AWS_REGION:-us-east-1}"
found=0

echo "== Resource Groups Tagging API (project=${PROJECT}) =="
# Two known sources of noise here, both benign (not billed, not a cleanup
# failure) - filtered out rather than left to cry wolf on every run:
#  1. ECS task-definition ARNs: AWS retains every revision forever, even
#     deregistered ones - there is no delete-able state to reach.
#  2. ECS cluster/service ARNs for a cluster already INACTIVE: ECS keeps
#     the ARN as inert metadata post-deletion; verified against the live
#     cluster status below rather than trusted at face value.
# The API itself also has a short eventual-consistency lag that can list an
# already-deleted resource (e.g. a subnet) for a few minutes - anything
# flagged here is cross-checked with a direct describe/list call, not
# taken on faith.
tagged=$(aws resourcegroupstaggingapi get-resources \
  --tag-filters "Key=project,Values=${PROJECT}" \
  --region "$REGION" \
  --query 'ResourceTagMappingList[].ResourceARN' --output text)

real_findings=""
for arn in $tagged; do
  case "$arn" in
    *:task-definition/*)
      continue
      ;;
    *:ecs:*:cluster/*|*:ecs:*:service/*)
      cluster_name=$(echo "$arn" | sed -E 's#.*(cluster|service)/([^/]+).*#\2#')
      status=$(aws ecs describe-clusters --clusters "$cluster_name" --region "$REGION" --query 'clusters[0].status' --output text 2>/dev/null || echo "MISSING")
      [ "$status" = "ACTIVE" ] && real_findings="$real_findings$arn (cluster status: ACTIVE)"$'\n'
      ;;
    *:ec2:*:subnet/*)
      subnet_id=$(echo "$arn" | sed -E 's#.*subnet/##')
      aws ec2 describe-subnets --subnet-ids "$subnet_id" --region "$REGION" >/dev/null 2>&1 && real_findings="$real_findings$arn"$'\n'
      ;;
    *)
      real_findings="$real_findings$arn"$'\n'
      ;;
  esac
done

if [ -n "$real_findings" ]; then
  printf '%s' "$real_findings"
  found=1
else
  echo "(none - or only benign AWS-retained metadata: INACTIVE ECS clusters/task-definition revisions)"
fi

echo "== ECS clusters named ${PROJECT}-* =="
clusters=$(aws ecs list-clusters --region "$REGION" --query "clusterArns[?contains(@, '${PROJECT}-')]" --output text)
if [ -n "$clusters" ]; then
  echo "$clusters"
  found=1
else
  echo "(none)"
fi

echo "== ECR repositories named ${PROJECT}-* =="
repos=$(aws ecr describe-repositories --region "$REGION" \
  --query "repositories[?starts_with(repositoryName, '${PROJECT}-')].repositoryArn" --output text 2>/dev/null || true)
if [ -n "$repos" ]; then
  echo "$repos"
  found=1
else
  echo "(none)"
fi

echo "== VPCs tagged Name=${PROJECT}-* =="
vpcs=$(aws ec2 describe-vpcs --region "$REGION" \
  --filters "Name=tag:Name,Values=${PROJECT}-*" \
  --query 'Vpcs[].VpcId' --output text)
if [ -n "$vpcs" ]; then
  echo "$vpcs"
  found=1
else
  echo "(none)"
fi

echo "== AWS Budgets named ${PROJECT}-* =="
account_id=$(aws sts get-caller-identity --query Account --output text)
budgets=$(aws budgets describe-budgets --account-id "$account_id" \
  --query "Budgets[?starts_with(BudgetName, '${PROJECT}-')].BudgetName" --output text)
if [ -n "$budgets" ]; then
  echo "$budgets"
  found=1
else
  echo "(none)"
fi

if [ "$found" -ne 0 ]; then
  echo
  echo "RESIDUAL RESOURCES FOUND - teardown is not complete."
  exit 1
fi

echo
echo "Clean - nothing tagged/named ${PROJECT} remains."
