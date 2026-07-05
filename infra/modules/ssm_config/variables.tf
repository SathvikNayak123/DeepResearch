variable "project" {
  type = string
}

variable "environment" {
  type = string
}

variable "plain_params" {
  description = "Non-secret config, one SSM String parameter per key."
  type        = map(string)
  default     = {}
}

variable "secret_params" {
  description = "Secret config (API keys, demo auth key), one SSM SecureString parameter per key. Not marked sensitive at the map level - for_each needs the key set unmarked; the individual values still carry sensitivity from their sensitive-marked source variables at the root."
  type        = map(string)
  default     = {}
}
