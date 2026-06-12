# Containerd v2.2 HTTP Registry Configuration Issue

## Problem
Containerd v2.2.2 (and the Transfer Service introduced in v2.1+) does not properly honor HTTP registry configurations even with:
- `use_local_image_pull = true` in CRI plugin config
- `config_path = '/etc/containerd/certs.d'` set correctly
- `hosts.toml` with `server = "http://..."` configuration

## Expected Configuration
```toml
# /etc/containerd/config.toml
version = 3

[plugins.'io.containerd.cri.v1.images']
  use_local_image_pull = true  # Should force client-side pulls
  
  [plugins.'io.containerd.cri.v1.images'.registry]
    config_path = "/etc/containerd/certs.d"

# /etc/containerd/certs.d/forge-reg.local:30500/hosts.toml
server = "http://forge-reg.local:30500"

[host."http://forge-reg.local:30500"]
  capabilities = ["pull", "resolve"]
```

## Actual Behavior
Despite the configuration, kubelet/crictl still tries to use HTTPS:
```
failed to do request: Head "https://forge-reg.local:30500/...": 
http: server gave HTTP response to HTTPS client
```

## Workarounds Tried
1. ✓ Set `use_local_image_pull = true` - Doesn't work in v2.2.2
2. ✓ Configure hosts.toml - Not honored by Transfer Service
3. ✓ Pre-pull images with ctr --plain-http - Works, but kubelet still tries HTTPS pull
4. ✓ Configure config_path in both CRI and Transfer Service - Doesn't help
5. ? Set imagePullPolicy: Never - Not feasible in RayService CRD

## Recommended IaC Solution
Use HTTPS for the registry instead of HTTP:

### 1. Generate self-signed certificate with SANs
```bash
cat > /tmp/cert.conf << 'CERT_CONF'
[req]
default_bits = 2048
prompt = no
default_md = sha256
req_extensions = v3_req
distinguished_name = dn

[dn]
C = US
ST = CA
L = San Francisco
O = Forge Registry
CN = forge-reg.local

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = forge-reg.local
DNS.2 = *.forge-reg.local
IP.1 = 192.168.86.184
IP.2 = 127.0.0.1
CERT_CONF

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /tmp/tls.key -out /tmp/tls.crt \
  -config /tmp/cert.conf -extensions v3_req
```

### 2. Create HTTPS proxy (or use native HTTPS registry)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: forge-registry-https
  namespace: infra
spec:
  replicas: 1
  selector:
    matchLabels:
      app: forge-registry-https
  template:
    metadata:
      labels:
        app: forge-registry-https
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
        ports:
        - containerPort: 30500
        volumeMounts:
        - name: config
          mountPath: /etc/nginx/nginx.conf
          subPath: nginx.conf
        - name: certs
          mountPath: /etc/nginx/certs
      volumes:
      - name: config
        configMap:
          name: nginx-registry-https
      - name: certs
        secret:
          secretName: forge-registry-certs
```

### 3. Trust the certificate
```bash
# Add to system trust store
sudo cp /tmp/tls.crt /usr/local/share/ca-certificates/forge-registry.crt
sudo update-ca-certificates

# Add to containerd
sudo mkdir -p /etc/containerd/certs.d/_default
sudo cp /tmp/tls.crt /etc/containerd/certs.d/_default/ca.crt
```

### 4. Use default containerd config
No special configuration needed - containerd v2.x handles HTTPS correctly by default.

## Conclusion
Containerd v2.x Transfer Service has a bug or limitation where it doesn't properly honor HTTP registry configurations. The cleanest IaC solution is to use HTTPS with proper certificates rather than fighting the configuration.
