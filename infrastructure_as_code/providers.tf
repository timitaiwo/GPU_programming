terraform {
  required_providers {
    runpod = {
      source = "decentralized-infrastructure/runpod"
      version = "1.0.1"
    }
  }
}

provider "runpod" {
  api_token = var.runpod_api_key
}
