output "alb_url" {
  description = "Live URL - POST {this}/research, GET {this}/health"
  value       = "http://${module.alb.alb_dns_name}"
}

output "ecr_repository_url" {
  value = module.ecr.repository_url
}

output "ecs_cluster_name" {
  value = module.ecs.cluster_name
}

output "ecs_service_name" {
  value = module.ecs.service_name
}

output "github_deploy_role_arn" {
  description = "Set as the GitHub Actions repo variable AWS_DEPLOY_ROLE_ARN (Settings → Secrets and variables → Actions → Variables). Empty when enable_github_oidc = false."
  value       = var.enable_github_oidc ? module.github_oidc[0].deploy_role_arn : ""
}
