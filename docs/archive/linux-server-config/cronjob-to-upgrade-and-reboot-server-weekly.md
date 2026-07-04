# Cronjob to upgrade and reboot server weekly

```bash
(sudo crontab -l 2>/dev/null | grep -v 'apt-get update.*upgrade'; echo '0 0 * * 0 apt-get update && apt-get upgrade -y && /sbin/shutdown -r now') | sudo crontab -
```
