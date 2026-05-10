# Tech Noir — Docker Bake configuration
#
# Build + push all images to Forge Registry:
#   docker bake --push
#
# Build only specific targets:
#   docker bake gpu-all --push
#
# Local build (no push):
#   docker bake
#
# Override registry:
#   REGISTRY=ghcr.io/jaydataengineer docker bake --push

variable "REGISTRY" {
  default = "forge-reg.local:30500/tech-noir"
}

group "default" {
  targets = ["ray-base", "gpu-all", "model-sync"]
}

target "ray-base" {
  dockerfile = "infra/docker/Dockerfile.base"
  context    = "."
  tags       = ["${REGISTRY}/ray-base:latest"]
  platforms  = ["linux/amd64"]
}

target "gpu-all" {
  dockerfile = "infra/docker/Dockerfile.gpu-all"
  context    = "."
  tags       = ["${REGISTRY}/gpu-all:latest"]
  platforms  = ["linux/amd64"]
}

target "model-sync" {
  dockerfile = "infra/docker/Dockerfile.model-sync"
  context    = "."
  tags       = ["${REGISTRY}/model-sync:latest"]
  platforms  = ["linux/amd64"]
}
