variable "project" {
  type = string
}

variable "environment" {
  type = string
}

variable "github_owner" {
  description = "GitHub org/user that owns the repo allowed to assume the deploy role."
  type        = string
}

variable "github_repo" {
  description = "Repo name (without owner) allowed to assume the deploy role."
  type        = string
}

variable "github_branch" {
  description = "Only workflow runs on this branch (ref) may assume the role. Scoped in the OIDC trust policy's `sub` condition — this is the least-privilege boundary that stops other repos/branches from deploying."
  type        = string
  default     = "main"
}

variable "create_oidc_provider" {
  description = "Create the account-level GitHub OIDC provider. Set false if the account already has one (only one per account is allowed) and pass its ARN via existing_oidc_provider_arn."
  type        = bool
  default     = true
}

variable "existing_oidc_provider_arn" {
  description = "ARN of a pre-existing GitHub OIDC provider, used only when create_oidc_provider = false."
  type        = string
  default     = ""
}

variable "ecr_repository_arn" {
  description = "ARN of the ECR repo the deploy role may push to (push actions scoped to exactly this repo)."
  type        = string
}

variable "ecs_cluster_name" {
  type = string
}

variable "ecs_service_name" {
  type = string
}

variable "execution_role_arn" {
  description = "Task execution role ARN — the deploy role may PassRole ONLY this and task_role_arn, and only to ecs-tasks.amazonaws.com (needed to register a new task definition revision)."
  type        = string
}

variable "task_role_arn" {
  type = string
}
