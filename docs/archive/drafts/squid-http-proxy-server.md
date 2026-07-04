# Run a Squid HTTP Proxy Server

```sh
sudo apt update && sudo apt install squid -y
```

```bash
#!/usr/bin/env bash
set -euo pipefail

# Parameters
SQUID_CONF="/etc/squid/squid.conf"
BACKUP_CONF="${SQUID_CONF}.orig_$(date +%Y%m%d_%H%M%S)"
HELPER="/usr/local/bin/squid_auth.sh"
LISTEN_PORT=3128

# Credentials (change these!)
PROXY_USER="username"
PROXY_PASS="password"

# 1) Install Squid only
apt-get update
apt-get install -y squid

# 2) Backup original config
cp "$SQUID_CONF" "$BACKUP_CONF"

# 3) Create a simple Bash authentication helper
echo "Creating authentication helper at $HELPER"
cat > "$HELPER" <<'EOF'
#!/usr/bin/env bash
set +u  # Disable unset-variable errors

# Read username and password from Squid
read user pass || exit 1

# Check credentials
if [[ "$user" == "username" && "$pass" == "password" ]]; then
  echo "OK"
else
  echo "ERR"
fi
EOF
chmod +x "$HELPER"

# 4) Generate minimal squid.conf allowing any IP, requiring only auth
cat > "$SQUID_CONF" <<EOF
# Squid minimal configuration
http_port $LISTEN_PORT

# Authentication setup
auth_param basic program $HELPER
auth_param basic realm Proxy
acl valid_user proxy_auth REQUIRED

# Access control and error message
deny_info ERR_AUTH_REQUIRED all
http_access allow valid_user
http_access deny all

# Logging
access_log /var/log/squid/access.log squid
cache_log /var/log/squid/cache.log
EOF

# 5) Restart & enable Squid
systemctl restart squid
systemctl enable squid

echo "✅ Squid configured on port $LISTEN_PORT for all IPs; authentication required with user '$PROXY_USER'."
echo "Test with: curl -v --proxy-user $PROXY_USER:$PROXY_PASS --proxy http://<server_ip>:$LISTEN_PORT https://ipinfo.io/"
```
