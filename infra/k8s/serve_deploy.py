"""Deploy serve applications on Ray cluster startup."""
import json, os, time

import requests
import yaml

CONFIG = """http_options:
  port: 8000
applications:
  - name: forge
    import_path: serve_config:forge
    route_prefix: /forge
    deployments:
      - name: forge
        autoscaling_config: { min_replicas: 0, max_replicas: 1, downscale_delay_s: 120 }
  - name: playground
    import_path: serve_config:playground
    route_prefix: /playground
    deployments:
      - name: playground
        autoscaling_config: { min_replicas: 0, max_replicas: 1, downscale_delay_s: 30 }
  - name: api-ingress
    import_path: serve_config:api_ingress
    route_prefix: /
    deployments:
      - name: api-ingress
        autoscaling_config: { min_replicas: 1, max_replicas: 1 }
"""


def main():
    dashboard = "http://localhost:8265"

    for i in range(30):
        try:
            r = requests.get(f"{dashboard}/api/serve/applications/", timeout=5)
            if r.status_code < 500:
                break
        except requests.ConnectionError:
            pass
        time.sleep(2)

    config = yaml.safe_load(CONFIG)
    resp = requests.put(
        f"{dashboard}/api/serve/applications/",
        json=config,
        timeout=30,
    )
    print(f"serve deploy: {resp.status_code}", flush=True)
    if resp.text:
        print(resp.text[:2000], flush=True)


if __name__ == "__main__":
    main()
