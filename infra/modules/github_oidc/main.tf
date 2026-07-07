# GitHub Actions OIDC → AWS, keyless CD.
#
# No long-lived AWS access keys anywhere: GitHub's OIDC provider mints a
# short-lived token per workflow run, and this role's trust policy only lets
# runs from exactly {github_owner}/{github_repo} on {github_branch} exchange
# that token for temporary AWS credentials. Rotating "keys" is not a thing
# that can leak here — there are none.

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# GitHub's well-known OIDC thumbprint list is no longer required by IAM (AWS
# validates the provider's TLS chain directly), but the resource still accepts
# it; leaving thumbprint_list unset is supported on current provider versions.
resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_oidc_provider ? 1 : 0

  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

locals {
  oidc_provider_arn = var.create_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : var.existing_oidc_provider_arn
  service_arn       = "arn:aws:ecs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:service/${var.ecs_cluster_name}/${var.ecs_service_name}"
}

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"

    principals {
      type        = "Federated"
      identifiers = [local.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # The load-bearing scope: only this repo, only this branch. A run from
    # any other repo/branch/PR presents a different `sub` and is denied.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_owner}/${var.github_repo}:ref:refs/heads/${var.github_branch}"]
    }
  }
}

resource "aws_iam_role" "deploy" {
  name                 = "${var.project}-${var.environment}-gha-deploy"
  assume_role_policy   = data.aws_iam_policy_document.assume.json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "deploy" {
  # ECR auth token has no resource-level scoping (AWS constraint) — the push
  # actions below ARE scoped to the one repo, so a token alone buys nothing.
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "EcrPushToOwnRepoOnly"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:PutImage",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [var.ecr_repository_arn]
  }

  # Roll a new deployment on exactly this one service. UpdateService/
  # DescribeServices are scoped to the service ARN.
  statement {
    sid       = "EcsDeployOwnServiceOnly"
    actions   = ["ecs:UpdateService", "ecs:DescribeServices"]
    resources = [local.service_arn]
  }

  # RegisterTaskDefinition / DescribeTaskDefinition / DescribeTasks have no
  # resource-level ARN support in IAM (AWS constraint) — must be "*".
  statement {
    sid       = "EcsTaskDefRegisterDescribe"
    actions   = ["ecs:RegisterTaskDefinition", "ecs:DescribeTaskDefinition", "ecs:DescribeTasks"]
    resources = ["*"]
  }

  # Registering a task-def revision requires passing the app's roles to ECS.
  # Scoped to exactly the two roles this task uses, and only to the ECS
  # tasks service — this role can't hand these roles to anything else.
  statement {
    sid       = "PassTaskRolesToEcsOnly"
    actions   = ["iam:PassRole"]
    resources = [var.execution_role_arn, var.task_role_arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "${var.project}-${var.environment}-gha-deploy"
  role   = aws_iam_role.deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}
