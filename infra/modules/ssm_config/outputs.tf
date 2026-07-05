output "plain_param_arns" {
  value = { for k, v in aws_ssm_parameter.plain : k => v.arn }
}

output "secret_param_arns" {
  value = { for k, v in aws_ssm_parameter.secret : k => v.arn }
}

output "param_path_prefix" {
  value = "/${var.project}/${var.environment}"
}
