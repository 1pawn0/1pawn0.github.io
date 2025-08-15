# 🌐 Setup Guide: Deploy Sphinx Docs to GitHub Pages

This guide walks you through setting up [`Sphinx`](https://www.sphinx-doc.org/) with Markdown support, and deploying your documentation to **GitHub Pages** using GitHub Actions.

---

## 🧱 Step 1: Create Your GitHub Repo

1. Go to GitHub and create a repo named:  

   ```
   your-github-username.github.io
   ```

2. Inside your actual project repo (e.g., `your-github-repo-name`), create the following file:  

   ```
   .github/workflows/main.yml
   ```

---

## ⚙️ Step 2: GitHub Actions Workflow for Sphinx

<details>
  <summary>📜 Click to view the GitHub Actions YAML</summary>

  <pre>
<code id="yaml-snippet">
name: Build and Deploy Sphinx Docs to GitHub Pages

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build_and_deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.13"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r docs/requirements.txt

      - name: Build Sphinx docs
        run: sphinx-build -b html docs/source docs/build/html

      - name: Configure Pages
        uses: actions/configure-pages@v5

      - name: Add .nojekyll
        run: touch docs/build/html/.nojekyll

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/build/html

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
        with:
          artifact_name: github-pages
          preview: false
</code>
  </pre>

  <button onclick="navigator.clipboard.writeText(document.getElementById('yaml-snippet').innerText)">📋 Copy YAML</button>
</details>

---

## 📦 Step 3: Install Sphinx and Markdown Extensions

Install Sphinx, MyST parser, and live preview tools in your conda environment:

```bash
conda install -y -c conda-forge sphinx myst-parser sphinx-autobuild
```

---

## 🌀 Step 4: Clone Your GitHub Pages Repo

```bash
gh repo clone your-github-username/your-github-username.github.io
cd your-github-username.github.io
```

---

## 🧙 Step 5: Create the Sphinx Project

Use `sphinx-quickstart` to initialize the docs folder:

```bash
sphinx-quickstart docs
```

Inside `docs/source/`, delete the default `index.rst` and replace it with a Markdown file named `index.md`:

````markdown
# Your Page Title

```{toctree}
:maxdepth: 2
:caption: Markdown Documents
:glob:

markdowns/*
```
````

Then, create the `docs/source/markdowns/` folder and drop all your Markdown content in there.

---

You're now ready to push your changes and let the GitHub Actions magic deploy your docs! 🚀  
