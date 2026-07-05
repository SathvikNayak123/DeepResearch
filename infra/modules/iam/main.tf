# Two distinct roles, deliberately not merged (docs/DESIGN.md: task
# execution role vs task role, least-privilege, task role gets SSM read +
# CloudWatch write and nothing else):
#
# - execution role: the ECS agent assumes this to pull the image from ECR,
#   resolve the SecureString `secrets` in the task def, and stand up the
#   awslogs driver. The app never assumes this role.
# - task role: the *app process* assumes this at runtime (via the task
#   metadata credentials endpoint) for any AWS SDK calls it makes itself.

data "aws_caller_identity" "current" {}

# --- Task execution role ---

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.project}-${var.environment}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Resolving SecureString `secrets` in the task def is a separate permission
# from the managed policy above (which only covers ECR pull + log creation).
data "aws_iam_policy_document" "execution_secrets" {
  count = length(var.secret_param_arns) > 0 ? 1 : 0

  statement {
    actions   = ["ssm:GetParameters"]
    resources = var.secret_param_arns
  }

  statement {
    actions   = ["kms:Decrypt"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${data.aws_region.current.name}.amazonaws.com"]
    }
  }
}

data "aws_region" "current" {}

resource "aws_iam_role_policy" "execution_secrets" {
  count  = length(var.secret_param_arns) > 0 ? 1 : 0
  name   = "${var.project}-${var.environment}-execution-secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_secrets[0].json
}

# --- Task role: SSM read + CloudWatch write, nothing else ---

data "aws_iam_policy_document" "task_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task" {
  name               = "${var.project}-${var.environment}-task"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
}

data "aws_iam_policy_document" "task_permissions" {
  statement {
    sid       = "SSMReadOwnPathOnly"
    actions   = ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"]
    resources = ["arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:parameter${var.ssm_param_path_prefix}/*"]
  }

  statement {
    sid       = "CloudWatchLogsWriteOwnGroupOnly"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["${var.log_group_arn}:*"]
  }

  statement {
    sid       = "CloudWatchCustomMetricsOwnNamespaceOnly"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"] # PutMetricData has no resource-level ARNs; scoped by condition below instead
    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = [var.metrics_namespace]
    }
  }
}

resource "aws_iam_role_policy" "task_permissions" {
  name   = "${var.project}-${var.environment}-task-permissions"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_permissions.json
}
