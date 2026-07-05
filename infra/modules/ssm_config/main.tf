# Config lives in SSM, not the image (docs/DESIGN.md universal rule: env via
# SSM, never baked in). Task def references these by name/ARN as
# `secrets`/`environment` - the execution role resolves SecureString values at
# launch (modules/iam), the app never sees the underlying KMS key.

resource "aws_ssm_parameter" "plain" {
  for_each = var.plain_params

  name  = "/${var.project}/${var.environment}/${each.key}"
  type  = "String"
  value = each.value
}

resource "aws_ssm_parameter" "secret" {
  for_each = var.secret_params

  name  = "/${var.project}/${var.environment}/${each.key}"
  type  = "SecureString"
  value = each.value
}
