# Containerd v2.2 Registry Solution

## Problem Statement
Containerd v2.2.2 (via the Transfer Service introduced in v2.1+) does not support:
- HTTP registries (use_local_image_pull is ignored)
- HTTPS with self-signed certificates (skip_verify in hosts.toml is ignored)
- Custom registry configurations of any kind

## Root Cause
The Transfer Service in containerd v2.x hardcodes HTTPS behavior and does not read
hosts.toml files or respect use_local_image_pull settings.

## Working Solutions

### Option 1: Use Public Registry
Push images to Docker Hub, GHCR, or another public registry with trusted certificates.
- Pros: Works out of the box
- Cons: Not suitable for local development, requires internet

### Option 2: Downgrade Containerd
Downgrade to containerd v1.x which properly supports HTTP registries.
```bash
# Ubuntu/Debian
sudo apt remove containerd
sudo apt install containerd.io=1.6.*
```
- Pros: HTTP works, proper configuration support
- Cons: Older version, potential compatibility issues

### Option 3: Upgrade Containerd
Upgrade to containerd v2.3+ which may have fixed Transfer Service issues.
- Pros: Latest features, potential fixes
- Cons: May still have issues, requires testing

### Option 4: HTTPS with Proper Certificate (Recommended for IaC)
Use HTTPS with a CA-signed certificate (not self-signed).

#### Steps:
1. Generate certificate with proper SANs
2. Add CA to system trust store
3. Configure registry with HTTPS
4. No special containerd configuration needed

```bash
# Generate certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -config <(cat <<'CERT_CONF'
[req]
default_bits = 2048
prompt = no
default_md = sha256
req_extensions = v3_req
distinguished_name = dn
[dn]
CN = forge-reg.local
[v3_req]
subjectAltName = DNS:forge-reg.local,DNS:*.forge-reg.local,IP:192.168.86.184
CERT_CONF
)

# Add to system trust
sudo cp tls.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates

# Use HTTPS registry
kubectl create secret tls registry-certs --cert=tls.crt --key=tls.key
# Deploy registry with HTTPS...
```

## Testing Status
Tested configurations that DON'T work with containerd v2.2.2:
- ✗ HTTP with use_local_image_pull = true
- ✗ HTTP with config_path and hosts.toml
- ✗ HTTPS with self-signed cert
- ✗ HTTPS with self-signed cert + skip_verify in hosts.toml
- ✗ Pre-pulled images with ctr (not visible to CRI)

## Recommendation
For a working IaC local development setup:
1. Use containerd v1.x for HTTP registry support, OR
2. Implement proper HTTPS with CA-signed certificate
