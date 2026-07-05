# Public-subnets-only design: ALB and Fargate tasks both sit in public
# subnets, tasks get a public IP directly (assign_public_ip = true on the
# service). No NAT gateway - it's a ~$32/mo fixed cost for outbound-only
# traffic (ECR pull, Anthropic/Tavily calls) that a public IP does for free.
# Tradeoff: task ENIs are internet-addressable; the security group is the
# only thing standing between them and the world, so it only opens the app
# port and only from the ALB's security group (see modules/ecs, modules/alb).

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "main" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "${var.project}-${var.environment}-vpc" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = { Name = "${var.project}-${var.environment}-igw" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = { Name = "${var.project}-${var.environment}-public-${count.index}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "${var.project}-${var.environment}-public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Isolated subnets (no IGW/NAT route) for the optional managed data layer
# (modules/data_managed, docs/DESIGN.md decision row 13). RDS/ElastiCache
# need no outbound internet access, so - unlike the app tasks - they belong
# in a subnet with no route out at all, not a public one. Free to create
# even when unused (var.use_managed_data_layer = false, the default).
resource "aws_subnet" "isolated" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = { Name = "${var.project}-${var.environment}-isolated-${count.index}" }
}
