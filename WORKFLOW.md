# Flux GitOps Workflow

Two branches, one cluster. Flux watches both and reconciles automatically.

## Branches

| Branch | Flux Source | What it controls |
|--------|-------------|------------------|
| `master` | `tech-noir` | Production — all services, infra, networking |
| `dev` | `tech-noir-dev` | Dev-only resources from `infra/flux/dev/` |

Both sources watch the same repo. Production reads from `master`, dev reads from `dev`. Push to either branch and Flux applies within 5 minutes.

## Daily Workflow

### Working on a feature

```bash
git checkout dev
# make changes anywhere in the repo
git push origin dev
```

If you added resources to `infra/flux/dev/`, Flux auto-deploys them. If you changed production configs (anything outside `infra/flux/dev/`), those won't take effect until you merge to master.

### Deploying to production

```bash
git checkout master
git merge dev
git push origin master
```

Flux reconciles within 5 minutes. Force it immediately:

```bash
flux reconcile source git tech-noir
flux reconcile kustomization <layer>
```

### Quick force-reconcile everything

```bash
flux reconcile source git tech-noir --with-source
flux reconcile source git tech-noir-dev --with-source
```

### Check what's deployed

```bash
flux get sources git        # both branches, last fetched revision
flux get kustomizations     # all 10 layers, current revision per branch
```

## Dev Resources

`infra/flux/dev/` on the `dev` branch is the dev playground. Anything there auto-deploys when you push. Current resources:

- **service-smoke-test** — CronJob pinging Ray, Grafana, Loki, Forge Registry every 10 minutes

Add more: test jobs, debug ConfigMaps, experimental manifests, one-off batch jobs. They're separate from production — no namespace conflicts.

```bash
# Add a new dev resource
vim infra/flux/dev/my-test.yaml
# Add it to the kustomization
vim infra/flux/dev/kustomization.yaml
git add infra/flux/dev/ && git commit -m "Add my test"
git push origin dev
```

## Flux Architecture

```
infra/flux/clusters/forge/kustomization.yaml   ← bootstrap point (apply manually)
├── tech-noir source (master branch)
│   ├── namespaces
│   ├── infra-storage
│   ├── infra-secrets (SOPS)
│   ├── helm (HelmRepository + HelmRelease)
│   ├── git (Gitea + runners)
│   ├── infra-services (postgres, vector, monitoring...)
│   ├── ai-services (Ray cluster)
│   ├── mcp (web-research, media-analysis)
│   └── networking (Traefik routes)
└── tech-noir-dev source (dev branch)
    └── dev-resources
        └── infra/flux/dev/ on dev branch
```

### Dependency order

Production layers apply sequentially:

```
namespaces → infra-storage → infra-secrets → helm → git + infra-services → ai-services + mcp → networking
```

`dev-resources` depends only on `namespaces`.

### Updating the bootstrap config

If you change `infra/flux/clusters/forge/kustomization.yaml` (add/remove sources or layers), Flux doesn't auto-update the bootstrap. Re-apply manually:

```bash
kubectl apply -f infra/flux/clusters/forge/kustomization.yaml
```

## Key Paths

| Path | Branch | Purpose |
|------|--------|---------|
| `infra/flux/clusters/forge/` | master | Flux bootstrap (GitRepositories + Kustomization layers) |
| `infra/flux/namespaces/` | master | Namespace definitions |
| `infra/flux/infra-storage/` | master | PVs and PVCs |
| `infra/flux/infra-secrets/` | master | SOPS-encrypted secrets |
| `infra/flux/helm/` | master | Helm repos + releases |
| `infra/flux/git/` | master | Gitea + act runners |
| `infra/flux/infra-services/` | master | Postgres, Vector, monitoring, Forge Registry |
| `infra/flux/ai-services/` | master | Ray cluster + Serve config |
| `infra/flux/mcp/` | master | MCP servers |
| `infra/flux/networking/` | master | Traefik ingress routes |
| `infra/flux/dev/` | dev | Dev-only resources (smoke tests, debug configs) |
| `infra/k8s/` | both | Build scripts + app code only (serve_config.py, Dockerfiles) |

## Troubleshooting

```bash
# Something not deploying?
flux get kustomizations                    # check revision and status
flux get kustomization <name> -o yaml      # full details + error message
flux reconcile kustomization <name>        # force retry

# Source not fetching?
flux get sources git                       # check last fetched revision
flux reconcile source git tech-noir        # force fetch
flux reconcile source git tech-noir-dev    # force fetch dev

# Check what Flux actually applied
kubectl get events -n flux-system --sort-by='.lastTimestamp'

# SOPS decryption failing?
kubectl get secret sops-age -n flux-system # key must exist
```
