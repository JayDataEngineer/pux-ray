# Containerd Image Pull Solution for K3s

## The Problem
After extensive troubleshooting, we discovered that **K3s uses its own bundled containerd** instead of the standalone containerd installed on the system. This is why all our earlier configuration of `/etc/containerd/config.toml` was not working.

## The Solution

### Step 1: Configure K3s Registry
Create `/etc/rancher/k3s/registries.yaml`:
```yaml
mirrors:
  forge-registry.infra.svc.cluster.local:5000:
    endpoints:
      - http://10.43.111.59:5000
configs:
  "forge-registry.infra.svc.cluster.local:5000":
    tls:
      insecure_skip_verify: true
```

### Step 2: Create ClusterIP Registry Service
```bash
kubectl create service clusterip forge-registry -n infra --tcp=5000
# Or use the YAML method with proper selector
```

### Step 3: Restart K3s
```bash
sudo systemctl restart k3s
```

## Current Status
- ✅ HTTP registry accessible via ClusterIP
- ✅ DNS resolves correctly  
- ✅ HTTP registry serves catalog successfully
- ❌ K3s containerd still uses HTTPS (not HTTP from config)
- ❌ K3s containerd caches old IP addresses

## Next Steps
The k3s containerd configuration appears to be caching or not properly loading the registry configuration. Possible solutions:
1. Use k3s containerd CLI to directly load configuration
2. Restart the entire node to clear all caches
3. Use a different registry approach (public registry, proper HTTPS)

## Files Modified
- `/etc/rancher/k3s/registries.yaml` - K3s registry configuration
- `/var/lib/rancher/k3s/agent/etc/containerd/certs.d/` - Host configuration files
- `/home/user/Documents/programs/ray/infra/k8s/forge-registry-service.yaml` - ClusterIP registry service

## Key Discovery
**K3s bundled containerd**: `/var/lib/rancher/k3s/.../bin/containerd`
**Socket**: `/run/k3s/containerd/containerd.sock`
**Config**: `/etc/rancher/k3s/registries.yaml`
