variable "project" {
  type = string
}

variable "environment" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  type = list(string)
}

variable "app_port" {
  type = number
}

variable "task_cpu" {
  type = number
}

variable "task_memory" {
  type = number
}

variable "ecr_repository_url" {
  type = string
}

variable "image_tag" {
  type = string
}

variable "execution_role_arn" {
  type = string
}

variable "task_role_arn" {
  type = string
}

variable "log_group_name" {
  type = string
}

variable "aws_region" {
  type = string
}

variable "alb_security_group_id" {
  type = string
}

variable "target_group_arn" {
  type = string
}

variable "alb_arn_suffix" {
  type = string
}

variable "target_group_arn_suffix" {
  type = string
}

variable "desired_count" {
  type = number
}

variable "min_capacity" {
  type = number
}

variable "max_capacity" {
  type = number
}

# docs/DESIGN.md decision row 12: request-count-per-target, not CPU - this
# workload is I/O-bound (waiting on Anthropic/Tavily), so CPU under-signals
# real load.
variable "requests_per_target_threshold" {
  type    = number
  default = 50
}

variable "app_secrets" {
  description = "Map of container env var name -> SSM parameter ARN, resolved by the execution role at launch."
  type        = map(string)
}

variable "include_data_containers" {
  description = "true = run Postgres + Redis as extra containers in this task (docs/DESIGN.md decision row 13, default path)."
  type        = bool
}

variable "postgres_password_ssm_arn" {
  type    = string
  default = ""
}
