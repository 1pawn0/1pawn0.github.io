# Cronjob to reboot server daily

```bash
(sudo crontab -l 2>/dev/null; echo '0 0 * * * /sbin/shutdown -r now') | sudo crontab -
```
