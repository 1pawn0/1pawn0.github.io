# Set up a SOCKS5 proxy server with `Dante`

Install `dante-server` on Ubuntu:
```sh
sudo apt update
sudo apt install dante-server
```

```sh
systemctl status danted.service
```

[Minimal server configuration](https://www.inet.no/dante/doc/1.4.x/config/server.html)

[IPv6 communication](https://www.inet.no/dante/doc/1.4.x/config/ipv6.html)

dante server config file path is: `/etc/danted.conf`
```sh
cat > /etc/danted.conf <<EOF
#logging
errorlog: /var/log/danted.errlog
logoutput: /var/log/danted.log

user.privileged: root
user.notprivileged: nobody

#server address specification
internal: 0.0.0.0 port = 32768
internal: :: port = 32768

external: eth0

#authentication methods
clientmethod: none
socksmethod: username

#accept connections from any client
client pass {
        from: 0/0 to: 0/0
        socksmethod: username
}

# Rule allowing authenticated SOCKS requests FROM the server (external interface)
# TO any destination IPv4 or IPv6 address.
# Authentication method must be 'username'.
socks pass {
        from: 0/0 to: 0/0
        socksmethod: username
}

EOF
```


Create user `nobody`
```sh
sudo useradd -r -s /sbin/nologin nobody
```

Create user `socksuser`
```sh
sudo useradd socksuser -s /sbin/nologin
sudo passwd socksuser
# Enter and confirm the password when prompted
```

start danted:
```sh
sudo systemctl enable --now danted
sudo systemctl restart danted
```

check danted status:
```sh
sudo systemctl status danted
sudo journalctl -u danted -f
```
