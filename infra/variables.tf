variable "aws_region" {
  description = "AWS region for the whole stack."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Short project name, used as a naming prefix and the tag every taggable resource carries (residual-check keys off this)."
  type        = string
  default     = "deepresearch"
}

variable "environment" {
  description = "Deployment environment name (demo/staging/prod). Kept single-value since this is a single-user portfolio deployment (docs/DESIGN.md non-goals)."
  type        = string
  default     = "demo"
}

variable "app_port" {
  description = "Port the FastAPI app listens on inside the container."
  type        = number
  default     = 8000
}

variable "image_tag" {
  description = "Tag of the image in ECR to deploy. scripts/deploy.sh pushes this tag and forces a new deployment."
  type        = string
  default     = "latest"
}

variable "desired_count" {
  description = "Baseline number of running tasks. Kept at 1 for a 2-10 user demo; autoscaling covers bursts."
  type        = number
  default     = 1
}

variable "min_capacity" {
  type    = number
  default = 1
}

variable "max_capacity" {
  description = "Autoscaling ceiling. Small on purpose - this is a cost-capped demo, not a fleet."
  type        = number
  default     = 2
}

variable "task_cpu" {
  description = "Total task-level CPU units (1024 = 1 vCPU), shared by the app + (if containers-in-task) postgres/redis containers."
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Total task-level memory in MiB."
  type        = number
  default     = 1024
}

variable "log_retention_days" {
  description = "CloudWatch log retention. Never infinite - this is a demo, not an audit trail."
  type        = number
  default     = 14
}

variable "use_managed_data_layer" {
  description = "true = RDS Postgres + ElastiCache Redis (infra/modules/data_managed). false (default) = Postgres/Redis run as extra containers in the same Fargate task. See docs/DESIGN.md decision row 13 for the cost math behind the default."
  type        = bool
  default     = false
}

variable "monthly_budget_limit_usd" {
  description = "AWS Budgets monthly limit. A manually-created console alarm at this same limit is a documented prerequisite completed before the *first* apply (chicken-and-egg: terraform's own alarm can't protect the apply that creates it). This resource keeps the alarm reproducible/torn-down alongside everything else from the second apply on."
  type        = number
  default     = 25
}

variable "budget_notification_email" {
  description = "Email address subscribed to the SNS topic that AWS Budgets alerts publish to."
  type        = string
}

variable "demo_api_key" {
  description = "Shared demo API key injected via SSM (DEEPRESEARCH_API_KEY). Not multi-tenant auth - a single gate on a public ALB URL that costs real LLM $ per call. Leave unset to disable the auth check."
  type        = string
  sensitive   = true
  default     = ""
}

variable "anthropic_api_key" {
  type      = string
  sensitive = true
  default   = ""
}

# The live deploy defaults to OpenRouter, not direct Anthropic - this
# project's own real-key runs (docs/RESULTS.md) hit Anthropic credit
# exhaustion repeatedly; OpenRouter/Gemini is the proven-working path. Set
# anthropic_api_key + flip deepresearch_llm_provider to "anthropic" (and the
# model vars below to Claude model ids) once Anthropic credit is available,
# with no code change needed either way - src/deepresearch/llm/client.py
# already branches on DEEPRESEARCH_LLM_PROVIDER.
variable "openrouter_api_key" {
  type      = string
  sensitive = true
  default   = ""
}

variable "deepresearch_llm_provider" {
  type    = string
  default = "openrouter"
}

variable "agent_model" {
  description = "Model id for planner/worker/reflection/synthesis - one var since this project's own sessions always run all four at the same tier."
  type        = string
  default     = "google/gemini-2.5-flash"
}

variable "judge_model" {
  type    = string
  default = "google/gemini-2.5-flash-lite"
}

# --- Keyless CD (GitHub Actions OIDC) ---

variable "enable_github_oidc" {
  description = "Create the GitHub OIDC provider + deploy role for keyless CD (.github/workflows/deploy.yml). Off by default so a plain apply needs no GitHub wiring."
  type        = bool
  default     = false
}

variable "github_owner" {
  description = "GitHub org/user owning the repo allowed to deploy."
  type        = string
  default     = "SathvikNayak123"
}

variable "github_repo" {
  description = "Repo name (without owner) allowed to deploy."
  type        = string
  default     = "DeepResearch"
}

variable "github_deploy_branch" {
  description = "Only this branch's workflow runs may assume the deploy role."
  type        = string
  default     = "main"
}

variable "create_github_oidc_provider" {
  description = "Create the account-level GitHub OIDC provider. Set false if the account already has one and pass existing_github_oidc_provider_arn."
  type        = bool
  default     = true
}

variable "existing_github_oidc_provider_arn" {
  description = "ARN of a pre-existing GitHub OIDC provider (only when create_github_oidc_provider = false)."
  type        = string
  default     = ""
}

variable "tavily_api_key" {
  type      = string
  sensitive = true
  default   = ""
}
