# mshimer.github.io

Personal blog built with [Hugo](https://gohugo.io/) and the [LoveIt](https://github.com/dillonzq/LoveIt) theme (git submodule).

## Local development

```bash
git clone --recurse-submodules git@github.com:mshimer/mshimer.github.io.git
brew install hugo          # extended edition required
hugo server -D             # http://localhost:1313
```

to generate the html locally :
```text
cd <your repo dir>
hugo --gc --minify
```

## New post

```bash
hugo new --kind posts posts/my-post.md
```

## Deploy

Pushing to `main` triggers `.github/workflows/hugo.yml`, which builds and publishes to GitHub Pages.
In the repo settings, set Pages > Source to "GitHub Actions".
