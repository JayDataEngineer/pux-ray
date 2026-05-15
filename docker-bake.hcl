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
  targets = ["gpu-all", "model-sync", "anigen", "postgres-age-vector", "mcp-web-research", "mcp-media-analysis"]
}

target "gpu-all" {
  dockerfile = "infra/docker/Dockerfile.gpu-all"
  context    = "."
  tags       = ["${REGISTRY}/gpu-all:latest", "${REGISTRY}/gpu-all:ngc-26.01"]
  platforms  = ["linux/amd64"]
}

target "model-sync" {
  dockerfile = "infra/docker/Dockerfile.model-sync"
  context    = "."
  tags       = ["${REGISTRY}/model-sync:latest"]
  platforms  = ["linux/amd64"]
}

target "anigen" {
  dockerfile = "infra/docker/Dockerfile.anigen"
  context    = "."
  tags       = ["${REGISTRY}/anigen:latest"]
  platforms  = ["linux/amd64"]
}

target "wan2gp" {
  dockerfile = "infra/docker/Dockerfile.wan2gp"
  context    = "."
  tags       = ["${REGISTRY}/wan2gp:latest"]
  platforms  = ["linux/amd64"]
}

target "postgres-age-vector" {
  dockerfile = "infra/docker/Dockerfile.postgres-age"
  context    = "."
  tags       = ["${REGISTRY}/postgres-age-vector:latest"]
  platforms  = ["linux/amd64"]
}

target "mcp-web-research" {
  dockerfile = "mcp/web-research/Dockerfile"
  context    = "mcp/web-research"
  tags       = ["${REGISTRY}/mcp-web-research:latest"]
  platforms  = ["linux/amd64"]
}

target "mcp-media-analysis" {
  dockerfile = "mcp/media-analysis/Dockerfile"
  context    = "mcp/media-analysis"
  tags       = ["${REGISTRY}/mcp-media-analysis:latest"]
  platforms  = ["linux/amd64"]
}
