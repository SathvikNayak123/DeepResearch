# Codifies the $25/mo budget alarm as IaC so it's reproducible and torn
# down/reapplied alongside everything else. The *first* apply relies on a
# manually-created console alarm at the same threshold instead (documented
# prerequisite, infra/README.md) - terraform's own alarm can't protect the
# apply that creates it. From the second apply on, this is the real one;
# the manual one can be deleted once this is confirmed working.

resource "aws_budgets_budget" "monthly" {
  name         = "${var.project}-${var.environment}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_limit_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = [format("user:project$%s", var.project)]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.notification_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.notification_email]
  }
}
