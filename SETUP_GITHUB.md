# Putting this on GitHub

A one-time, five-minute setup. Run these from inside the `flow-surrogates` folder.

## 1. Tell Git who you are (first time only)

```bash
git config --global user.name  "Balla Ashok"
git config --global user.email "your-email@example.com"
```

## 2. Create the repository locally

```bash
git init
git add .
git commit -m "Navier-Stokes solvers, reduced-order models and surrogates"
```

`.gitignore` already excludes the `.npz` data and `.pt` model files, so only code,
tests, figures and documentation are committed. Anyone can regenerate the data by
running the scripts.

## 3. Create it on GitHub

On github.com: **New repository** → name it `flow-surrogates` → **Public** →
do *not* tick "add a README" (you already have one) → **Create**.

Then connect and push:

```bash
git remote add origin https://github.com/YOUR-USERNAME/flow-surrogates.git
git branch -M main
git push -u origin main
```

## 4. Check it

- The README should render on the front page with the validation figure.
- Under the **Actions** tab, the test workflow should run and show a green tick.
  That green tick is worth having: it shows the tests pass on a clean machine,
  not just yours.

## 5. Use it

Add the link to your CV header, next to LinkedIn, and mention it in interviews.

## Working with it afterwards

```bash
git status                      # what has changed
git add -A                      # stage everything
git commit -m "describe it"     # save a snapshot
git push                        # send it to GitHub
```

Commit whenever something works, not only when everything is finished. A history
of small, described commits is itself evidence of good practice.
