# Run a SOCKS5 Proxy Server on Ubuntu

To run a SOCKS5 proxy server on Ubuntu, you can use the `microsocks` package.

```shell
sudo apt update
sudo apt install microsocks
```

Run the SOCKS5 server with the desired port and authentication:

```shell
nohup microsocks -p 1080 -u replace-it-with-your-username -P replace-it-with-your-password > /dev/null 2>&1 &

```

Now you can connect to the SOCKS5 proxy server from your client using the the server IP address or server domain and the port you specified:

```shell
!curl -x socks5h://replace-it-with-your-username:replace-it-with-your-password@server.public.ip.or.domain:1080 ipinfo.io
```
