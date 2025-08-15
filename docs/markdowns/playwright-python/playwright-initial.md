# Web Scraping via Async Playwright Python

[https://playwright.dev/python/docs/intro](https://playwright.dev/python/docs/intro)
[https://playwright.dev/python/docs/api/class-playwright](https://playwright.dev/python/docs/api/class-playwright)

## Install Playwright and Playwright Stealth

```sh
pip install playwright playwright-stealth
playwright install --with-deps chromium
```

## Import Required Libraries

```python
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright, Playwright, Browser, BrowserContext, Page
from playwright_stealth import Stealth
from tqdm.notebook import tqdm
from yarl import URL
```

## Setup HTTP Proxy

```python
http_proxy_ip = 'your.proxy.server.ip' # change it
http_proxy_port = '12345' # change it
http_proxy_username = 'username' # change it
http_proxy_password = 'password' # change it
proxy_server: dict[str, str] = {"server": f"{http_proxy_ip}:{http_proxy_port}", "username": http_proxy_username, "password": http_proxy_password}
```

## Set the path of storage state json file

```python
storage_state_path = Path('path/to/storage_state.json') # change it
```

## Initialize Playwright and Create a Page Instance

```python
from playwright.async_api import async_playwright, Playwright, Browser, BrowserContext, Page
from playwright_stealth import Stealth

# Initialize Page
context_manager = Stealth().use_async(async_playwright())
p: Playwright = await context_manager.__aenter__()
browser: Browser = await p.chromium.launch(channel='chromium', proxy=proxy_server)
context: BrowserContext = await browser.new_context(
    base_url='https://base-url-of-scraping-website.com/', # change it
    storage_state=storage_state_path,
    # java_script_enabled=False,
    )
page: Page = await context.new_page()
await page.goto('https://base-url-of-scraping-website.com/path',
    wait_until='networkidle',  # 'load', 'domcontentloaded', 'networkidle'
    timeout=10000,  # 10 seconds
) # change it
```

## Scrape Data

```python
texts = await page.locator('enter-css-selector-or-xpath').all_inner_texts()  # change it
```

## Save Storage State

```python
storage_state = await context.storage_state(path=storage_state_path, indexed_db=True)
```
