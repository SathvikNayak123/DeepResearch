variable "project" {
  type = string
}

variable "environment" {
  type = string
}

variable "log_group_arn" {
  type = string
}

variable "ssm_param_path_prefix" {
  description = "e.g. /deepresearch/demo - task role gets read scoped to this path only."
  type        = string
}

variable "secret_param_arns" {
  description = "SecureString parameter ARNs referenced as task-def `secrets` - execution role needs to resolve exactly these at launch, nothing broader."
  type        = list(string)
  default     = []
}

variable "metrics_namespace" {
  type    = string
  default = "DeepResearch"
}
