# infra — AWS deployment (Terraform)

ECS Fargate behind an ALB. See `docs/DESIGN.md` decision rows 12-13 for why
Fargate over EKS/EC2 and why containers-in-task over RDS/ElastiCache by
default. This doc is the how; that doc is the why.

## Prerequisites (manual, before the first `apply`)

1. AWS account + credentials configured (`aws sts get-caller-identity` works).
2. **A $25/mo AWS Budgets alarm, created manually in the console, before the
   first apply.** This Terraform also codifies that same alarm
   (`modules/budget`), but it can't protect the apply that creates it —
   the manual one is the actual guardrail for apply #1. Once the
   Terraform-managed one exists and is confirmed, the manual one can be
   deleted.
3. Terraform >= 1.5, AWS CLI v2, Docker (to build/push the image).
4. Copy `terraform.tfvars.example` to `terraform.tfvars` and fill in real
   values. **Never commit `terraform.tfvars`** — it holds API keys and the
   demo auth key (already gitignored).

## IAM for a non-root apply

If applying as a scoped IAM user/role rather than root, a hand-written
least-privilege policy will be missing some read-back/tagging actions the
first time it's exercised end-to-end — several AWS actions (tag reads,
lifecycle-policy reads, `ssm:DescribeParameters`, `ec2:*VpcAttribute`) aren't
obvious from the resources alone. Discovered by running this exact config
against a fresh custom policy (2026-07-04):

- ECR: `ecr:PutLifecyclePolicy`, `ecr:GetLifecyclePolicy`, `ecr:DeleteLifecyclePolicy`
- CloudWatch Logs: `logs:ListTagsForResource`
- Budgets: `budgets:TagResource`, `budgets:ListTagsForResource`
- EC2: `ec2:ModifyVpcAttribute`, `ec2:DescribeVpcAttribute`, `ec2:ModifySubnetAttribute`
- SSM: `ssm:DescribeParameters` — **must** be `Resource: "*"`, this action has
  no resource-level scoping (same AWS constraint as `cloudwatch:PutMetricData`)
- Application Auto Scaling: `application-autoscaling:TagResource`,
  `application-autoscaling:ListTagsForResource`, `application-autoscaling:DeleteScalingPolicy`

## Cost shape

At `desired_count = 1`, containers-in-task (default): ALB (~$16.4/mo fixed)
+ one 0.5 vCPU/1GB Fargate task (~$18/mo if left running continuously) is
already most of a $25/mo budget. **The intended usage pattern is
apply → verify the live URL → destroy**, not "leave it running all month" —
real spend for an apply/verify/destroy cycle is hours, i.e. cents. If you
want continuous uptime, budget for it explicitly (drop `desired_count`
further isn't possible below 1; the real lever is deciding whether the demo
needs to be up 24/7 at all).

## Apply

```sh
cd infra
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

Then build and push the image, and force a fresh deployment:

```sh
../scripts/deploy.sh
```

`scripts/deploy.sh` reads the ECR repo URL from `terraform output`, so it
must be run after `apply`.

Verify:

```sh
curl http://$(terraform output -raw alb_url | sed 's#http://##')/health
curl -X POST http://$(terraform output -raw alb_url | sed 's#http://##')/research \
  -H "X-API-Key: <your demo_api_key>" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

## Destroy + residual check

```sh
terraform destroy
../scripts/residual_check.sh
```

`residual_check.sh` queries ECS/ECR/ALB/RDS/ElastiCache/IAM/SSM/CloudWatch/
Budgets for anything still tagged `project=deepresearch` after destroy and
exits non-zero if it finds any — `terraform destroy` succeeding is not by
itself proof nothing was left behind (a resource outside terraform's state,
or a destroy that silently no-ops on one resource, wouldn't show up in its
exit code).
