resource "aws_ecs_cluster" "main" {
  name = "${var.project}-${var.environment}"
}

resource "aws_security_group" "tasks" {
  name        = "${var.project}-${var.environment}-tasks"
  description = "ECS tasks: app port from the ALB only. Postgres/Redis containers share the task ENI (localhost) and are never opened here - not reachable from anywhere but the app container."
  vpc_id      = var.vpc_id

  ingress {
    from_port       = var.app_port
    to_port         = var.app_port
    protocol        = "tcp"
    security_groups = [var.alb_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

locals {
  # Split task-level cpu/memory across containers only when Postgres/Redis
  # run in-task; otherwise the app container floats to the whole task
  # allocation (no per-container limit needed with a single container).
  app_cpu     = var.include_data_containers ? floor(var.task_cpu * 0.5) : null
  app_memory  = var.include_data_containers ? floor(var.task_memory * 0.5) : null
  side_cpu    = var.include_data_containers ? floor(var.task_cpu * 0.25) : null
  side_memory = var.include_data_containers ? floor(var.task_memory * 0.25) : null

  # All three container objects share an identical shape (empty list/null
  # for whatever doesn't apply) so the conditional list below type-checks -
  # HCL requires consistent element types across a single list value.
  app_container = {
    name         = "app"
    image        = "${var.ecr_repository_url}:${var.image_tag}"
    essential    = true
    cpu          = local.app_cpu
    memory       = local.app_memory
    portMappings = [{ containerPort = var.app_port, protocol = "tcp" }]
    environment  = []
    secrets = [
      for name, arn in var.app_secrets : { name = name, valueFrom = arn }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = var.log_group_name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "app"
      }
    }
  }

  postgres_container = {
    name         = "postgres"
    image        = "postgres:16-alpine"
    essential    = true
    cpu          = local.side_cpu
    memory       = local.side_memory
    portMappings = []
    environment = [
      { name = "POSTGRES_USER", value = "deepresearch" },
      { name = "POSTGRES_DB", value = "deepresearch" },
    ]
    secrets = [
      { name = "POSTGRES_PASSWORD", valueFrom = var.postgres_password_ssm_arn }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = var.log_group_name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "postgres"
      }
    }
  }

  redis_container = {
    name         = "redis"
    image        = "redis:7-alpine"
    essential    = true
    cpu          = local.side_cpu
    memory       = local.side_memory
    portMappings = []
    environment  = []
    secrets      = []
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = var.log_group_name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "redis"
      }
    }
  }

  container_definitions = var.include_data_containers ? [
    local.app_container, local.postgres_container, local.redis_container
    ] : [
    local.app_container
  ]
}

resource "aws_ecs_task_definition" "app" {
  family                   = "${var.project}-${var.environment}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = var.execution_role_arn
  task_role_arn            = var.task_role_arn

  container_definitions = jsonencode(local.container_definitions)
}

resource "aws_ecs_service" "app" {
  name            = "${var.project}-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = true # no NAT gateway (docs/DESIGN.md) - tasks need a public IP to pull images / reach the internet
  }

  load_balancer {
    target_group_arn = var.target_group_arn
    container_name   = "app"
    container_port   = var.app_port
  }

  # Let autoscaling own desired_count after the first apply - target
  # tracking below adjusts it, and re-applying with a stale desired_count
  # would otherwise fight the scaling policy.
  lifecycle {
    ignore_changes = [desired_count]
  }
}

resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

# ALBRequestCountPerTarget, not CPU (docs/DESIGN.md decision row 12 rationale
# in this module's variables.tf): each request is I/O-bound, waiting on
# Anthropic/Tavily, so CPU never climbs high enough to trigger a CPU-based
# policy under real concurrent load.
resource "aws_appautoscaling_policy" "requests" {
  name               = "${var.project}-${var.environment}-requests-per-target"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${var.alb_arn_suffix}/${var.target_group_arn_suffix}"
    }
    target_value       = var.requests_per_target_threshold
    scale_in_cooldown  = 60
    scale_out_cooldown = 60
  }
}
