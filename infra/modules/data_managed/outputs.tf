output "database_url" {
  value     = "postgresql+asyncpg://deepresearch:${var.db_password}@${aws_db_instance.postgres.address}:5432/deepresearch"
  sensitive = true
}

output "redis_url" {
  value = "redis://${aws_elasticache_cluster.redis.cache_nodes[0].address}:6379/0"
}
