# Only instantiated when var.use_managed_data_layer = true at the root
# (docs/DESIGN.md decision row 13). Off by default: RDS db.t4g.micro
# (~$12-13/mo) + ElastiCache cache.t4g.micro (~$12/mo) would roughly double
# the always-on cost of the ALB + Fargate task alone, which is already most
# of the $25/mo demo budget. Written and wired so flipping the variable is
# the only change needed once continuous uptime justifies the cost.

resource "aws_security_group" "data" {
  name        = "${var.project}-${var.environment}-data"
  description = "RDS + ElastiCache: reachable only from the ECS task SG"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.tasks_security_group_id]
  }

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [var.tasks_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.project}-${var.environment}-db"
  subnet_ids = var.isolated_subnet_ids
}

resource "aws_db_instance" "postgres" {
  identifier              = "${var.project}-${var.environment}"
  engine                  = "postgres"
  engine_version          = "16"
  instance_class          = var.db_instance_class
  allocated_storage       = 20
  db_name                 = "deepresearch"
  username                = "deepresearch"
  password                = var.db_password
  db_subnet_group_name    = aws_db_subnet_group.main.name
  vpc_security_group_ids  = [aws_security_group.data.id]
  publicly_accessible     = false
  skip_final_snapshot     = true # demo/teardown-friendly - no snapshot cost/lingering resource on destroy
  backup_retention_period = 0
}

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project}-${var.environment}-cache"
  subnet_ids = var.isolated_subnet_ids
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id         = "${var.project}-${var.environment}"
  engine             = "redis"
  engine_version     = "7.1"
  node_type          = var.redis_node_type
  num_cache_nodes    = 1
  port               = 6379
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.data.id]
}
