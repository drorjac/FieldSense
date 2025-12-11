# Contributing Guide

## Overview

This project uses a **fork workflow** with two main operations:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   PULL (common)                       PUSH (occasional)            │
│   Get updates from main repo          Submit your work for review  │
│          
                                                       │
│   Main Repo ────────────────> Your Fork ────────────> Main Repo │
│   (upstream)          sync       (origin)       PR              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| Direction | When | How Often |
|-----------|------|-----------|
| **PULL** (upstream → you) | Main repo updates `core/` or `datasets/` | Regular |
| **PUSH** (you → upstream) | Submit your work or suggest improvements | Occasional |

---

## Initial Setup (One Time)

### Step 1: Fork the Repository

1. Go to the GitHub repo page
2. Click the **Fork** button (top right)
3. This creates your own copy at `github.com/YOUR_USERNAME/FieldSense`

### Step 2: Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/FieldSense.git
cd FieldSense
```

### Step 3: Connect to Upstream

This lets you pull updates from the main repo:

```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/FieldSense.git
```

Verify your remotes:
```bash
git remote -v
# Should show:
# origin    https://github.com/YOUR_USERNAME/FieldSense.git (your fork)
# upstream  https://github.com/ORIGINAL_OWNER/FieldSense.git (main repo)
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
pip install -r projects/YOUR_PROJECT/requirements.txt
```

---

## PULL: Getting Updates from Main Repo

### When to Pull

- Before starting new work
- When there are updates to `core/` or `datasets/`
- If your code has errors that might be fixed upstream

### How to Pull

```bash
# 1. Fetch latest from main repo
git fetch upstream

# 2. Make sure you're on main
git checkout main

# 3. Merge the updates into your local copy
git merge upstream/main

# 4. Push to your fork (keeps your GitHub fork in sync)
git push origin main
```

### ⚠️ Important: Avoid Conflicts

To prevent merge conflicts when pulling:

- **Don't edit files in `core/`** — only add new files or suggest changes via PR
- **Don't edit files in `datasets/`** — same as above
- **Work only in `projects/YOUR_PROJECT/`** — this is your safe space

If you edit shared files locally, you may get conflicts when the main repo updates them.

---

## PUSH: Submitting Your Work

There are two types of submissions:

### Type A: Submitting Your Project Work (Most Common)

When you want to submit your notebooks, code, or results for review:

**Step 1: Save your work locally**
```bash
git add .
git commit -m "added analysis notebook for experiment X"
git push origin main
```

**Step 2: Open a Pull Request**
1. Go to the **main repo** on GitHub
2. Click **Pull Requests** → **New Pull Request**
3. Click **compare across forks**
4. Set:
   - **base repository:** `ORIGINAL_OWNER/FieldSense` | **base:** `main`
   - **head repository:** `YOUR_USERNAME/FieldSense` | **compare:** `main`
5. Add a title and description
6. Click **Create Pull Request**

### Type B: Suggesting Improvements to Core (Rare)

If you wrote something useful that could benefit everyone:

1. **Don't overwrite existing files** — instead:
   - Add a **new file** (e.g., `core/my_new_utils.py`)
   - Or create a **copy** with your changes (e.g., `core/plotting_v2.py`)
2. Commit and push to your fork
3. Open a Pull Request (same steps as above)
4. In the PR description, explain what you added and why

The maintainer will review, and if approved, merge it and possibly rename/reorganize.

---

## Daily Workflow Summary

```bash
# Start of work session - sync first
git fetch upstream
git checkout main
git merge upstream/main
git push origin main

# Do your work in projects/YOUR_PROJECT/
# ... edit notebooks, add code ...

# End of work session - save your work
git add .
git commit -m "description of what you did"
git push origin main

# When ready for review - open a PR on GitHub
```

---

## Repository Rules

### ✅ Your Space (Work Freely)
- `projects/YOUR_PROJECT/` — add and edit anything here

### 🔒 Shared Space (Pull Only, Don't Edit)
- `core/` — pull updates, but don't edit existing files
- `datasets/` — pull updates, but don't edit existing files

If you want to improve shared code, **add new files** and submit a PR.

### ❌ Off Limits
- Other projects' folders

---

## Project Structure

Keep your project organized:

```
projects/your-project/
├── src/              # Source code
├── notebooks/        # Jupyter notebooks
├── results/          # Outputs, figures, models
├── requirements.txt  # Project-specific dependencies
└── README.md         # Project documentation
```

---

## Troubleshooting

### Merge Conflicts

If you see this in a file:
```
<<<<<<< HEAD
your version
=======
upstream version
>>>>>>> upstream/main
```

**Option 1: Keep upstream version (recommended for shared files)**
```bash
git checkout --theirs filename
git add filename
git commit -m "resolved conflict, kept upstream version"
```

**Option 2: Keep your version**
```bash
git checkout --ours filename
git add filename
git commit -m "resolved conflict, kept my version"
```

**Option 3: Manual merge**
Edit the file, remove the markers, keep what you need, then:
```bash
git add filename
git commit -m "resolved conflict manually"
```

### Undo Local Changes (Before Commit)

```bash
git checkout -- filename
```

### Reset to Match Upstream (Last Resort)

⚠️ This discards ALL your uncommitted local changes:
```bash
git fetch upstream
git reset --hard upstream/main
git push origin main --force
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Check status | `git status` |
| Pull from upstream | `git fetch upstream && git merge upstream/main` |
| Save your work | `git add . && git commit -m "message"` |
| Push to your fork | `git push origin main` |
| View remotes | `git remote -v` |

---

## Need Help?

- Git basics: [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- Stuck? Ask before using `--force` commands!