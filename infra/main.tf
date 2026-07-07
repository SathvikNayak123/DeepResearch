module "networking" {
  source      = "./modules/networking"
  project     = var.project
  environment = var.environment
}

module "ecr" {
  source      = "./modules/ecr"
  project     = var.project
  environment = var.environment
}

module "logs" {
  source         = "./modules/logs"
  project        = var.project
  environment    = var.environment
  retention_days = var.log_retention_days
}

module "alb" {
  source      = "./modules/alb"
  project     = var.project
  environment = var.environment
  vpc_id      = module.networking.vpc_id
  subnet_ids  = module.networking.public_subnet_ids
  app_port    = var.app_port
}

# Postgres password: generated once, stored only in SSM (SecureString) and
# terraform state - never in the image, never in plain tfvars.
resource "random_password" "postgres" {
  length  = 24
  special = false
}

locals {
  ssm_prefix = "/${var.project}/${var.environment}"

  # Containers-in-task DATABASE_URL/REDIS_URL point at localhost (same task
  # ENI); managed-data-layer URLs come from modules.data_managed instead.
  # Only one of these is ever used, gated by var.use_managed_data_layer.
  database_url = var.use_managed_data_layer ? module.data_managed[0].database_url : "postgresql+asyncpg://deepresearch:${random_password.postgres.result}@localhost:5432/deepresearch"
  redis_url    = var.use_managed_data_layer ? module.data_managed[0].redis_url : "redis://localhost:6379/0"

  secret_params = {
    ANTHROPIC_API_KEY    = var.anthropic_api_key
    TAVILY_API_KEY       = var.tavily_api_key
    DEEPRESEARCH_API_KEY = var.demo_api_key
    DATABASE_URL         = local.database_url
    REDIS_URL            = local.redis_url
    POSTGRES_PASSWORD    = random_password.postgres.result
  }
}

module "ssm_config" {
  source        = "./modules/ssm_config"
  project       = var.project
  environment   = var.environment
  secret_params = local.secret_params
}

module "iam" {
  source                = "./modules/iam"
  project               = var.project
  environment           = var.environment
  log_group_arn         = module.logs.log_group_arn
  ssm_param_path_prefix = local.ssm_prefix
  secret_param_arns     = values(module.ssm_config.secret_param_arns)
}

module "ecs" {
  source      = "./modules/ecs"
  project     = var.project
  environment = var.environment
  vpc_id      = module.networking.vpc_id
  subnet_ids  = module.networking.public_subnet_ids
  aws_region  = var.aws_region
  app_port    = var.app_port
  task_cpu    = var.task_cpu
  task_memory = var.task_memory

  ecr_repository_url = module.ecr.repository_url
  image_tag          = var.image_tag

  execution_role_arn = module.iam.execution_role_arn
  task_role_arn      = module.iam.task_role_arn
  log_group_name     = module.logs.log_group_name

  alb_security_group_id   = module.alb.security_group_id
  target_group_arn        = module.alb.target_group_arn
  alb_arn_suffix          = module.alb.alb_arn_suffix
  target_group_arn_suffix = module.alb.target_group_arn_suffix

  desired_count = var.desired_count
  min_capacity  = var.min_capacity
  max_capacity  = var.max_capacity

  # App container only needs the SSM-backed secrets that vary the app's own
  # runtime config; POSTGRES_PASSWORD is consumed by the postgres sidecar,
  # not the app, so it's excluded here.
  app_secrets = {
    for k, arn in module.ssm_config.secret_param_arns : k => arn
    if k != "POSTGRES_PASSWORD"
  }

  include_data_containers   = !var.use_managed_data_layer
  postgres_password_ssm_arn = var.use_managed_data_layer ? "" : module.ssm_config.secret_param_arns["POSTGRES_PASSWORD"]
}

# Off by default (docs/DESIGN.md decision row 13) - written and wired so
# flipping var.use_managed_data_layer is the only change needed.
module "data_managed" {
  count       = var.use_managed_data_layer ? 1 : 0
  source      = "./modules/data_managed"
  project     = var.project
  environment = var.environment

  vpc_id                  = module.networking.vpc_id
  isolated_subnet_ids     = module.networking.isolated_subnet_ids
  tasks_security_group_id = module.ecs.tasks_security_group_id
  db_password             = random_password.postgres.result
}

# Keyless CD: GitHub Actions assumes this role via OIDC (no AWS keys in
# GitHub secrets). Trust scoped to this repo + branch only. Off by default so
# a plain `apply` needs no GitHub wiring; flip enable_github_oidc = true.
module "github_oidc" {
  count       = var.enable_github_oidc ? 1 : 0
  source      = "./modules/github_oidc"
  project     = var.project
  environment = var.environment

  github_owner               = var.github_owner
  github_repo                = var.github_repo
  github_branch              = var.github_deploy_branch
  create_oidc_provider       = var.create_github_oidc_provider
  existing_oidc_provider_arn = var.existing_github_oidc_provider_arn

  ecr_repository_arn = module.ecr.repository_arn
  ecs_cluster_name   = module.ecs.cluster_name
  ecs_service_name   = module.ecs.service_name
  execution_role_arn = module.iam.execution_role_arn
  task_role_arn      = module.iam.task_role_arn
}

module "budget" {
  source             = "./modules/budget"
  project            = var.project
  environment        = var.environment
  monthly_limit_usd  = var.monthly_budget_limit_usd
  notification_email = var.budget_notification_email
}
