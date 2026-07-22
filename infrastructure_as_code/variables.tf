variable "runpod_api_key" {
  type = string
  sensitive = true
}

variable "h100_type"{
    description = "The gpu type e.g H100"
    type = string
    default = "h100"
} 

variable "h100_template" {
  description = "Template to apply to the gpu pod"
  type = string
  default = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
}

