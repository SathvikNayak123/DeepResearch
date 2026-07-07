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

## Keyless CD (GitHub Actions OIDC)

`.github/workflows/deploy.yml` builds, pushes, and rolls a new ECS deployment
on every push to `main`, then smoke-checks the live ALB and **auto-rolls-back**
to the previous task-definition revision if the new one fails to stabilize or
fails `/health`. **No long-lived AWS keys exist anywhere** — GitHub mints a
short-lived OIDC token per run and exchanges it for the scoped deploy role
(`modules/github_oidc`), whose trust policy only accepts runs from *this repo
on the configured branch* (the `sub` condition). Do **not** put
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in GitHub secrets — OIDC replaces
them; storing them would reintroduce exactly the long-lived key this design
removes.

One-time setup:

1. Enable it in `terraform.tfvars`: `enable_github_oidc = true` (set
   `create_github_oidc_provider = false` + `existing_github_oidc_provider_arn`
   if the account already has a GitHub OIDC provider — only one is allowed per
   account), then `terraform apply`.
2. Set two **repo Actions *variables*** (Settings → Secrets and variables →
   Actions → *Variables* tab, not *Secrets* — neither value is sensitive,
   so Variables is the correct home): `AWS_DEPLOY_ROLE_ARN` and `ALB_URL`,
   both from `terraform output` (see below). `deploy.yml` reads Variables
   first and falls back to Secrets of the same name if you set them there
   instead, so either works — but prefer Variables, since anything in
   Secrets is treated (and masked in logs) as if it were sensitive, which
   these two values aren't.
   ```sh
   terraform output -raw github_deploy_role_arn   # -> AWS_DEPLOY_ROLE_ARN
   terraform output -raw alb_url                   # -> ALB_URL
   ```
   Both outputs only exist after a **full** `apply` of `infra/` (the ALB is
   part of every apply, not something `enable_github_oidc` gates) — if
   either command errors, run `terraform apply` first.
3. Push to `main` (or run the workflow manually via the Actions tab —
   `deploy.yml` has `workflow_dispatch` enabled) — the deploy is hands-free
   from there.

The deploy role is least-privilege: ECR push scoped to the one repo, ECS
`UpdateService`/`DescribeServices` scoped to the one service, and `iam:PassRole`
scoped to exactly the task's execution + task roles and only to
`ecs-tasks.amazonaws.com`. The only `Resource: "*"` entries
(`ecr:GetAuthorizationToken`, `ecs:RegisterTaskDefinition`/`DescribeTaskDefinition`)
are AWS actions with no resource-level ARN support.

To capture the two evidence artifacts the design calls for: a clean push to
`main` gives you **one hands-free deploy** run; pushing a commit whose image
fails `/health` (e.g. a bad `CMD`) gives you **one auto-rollback** run (job goes
red, service returns to the prior revision) — link both run URLs in
`docs/RESULTS.md`.
