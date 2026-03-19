# Deploying Vrite to Render

---

## Before you start

Make sure when you unzip Vrite.zip the files sit at the ROOT of your
repo, not inside a subfolder. Your repo on GitHub must look like this:

    your-repo/
    |-- Dockerfile        <- must be visible here on the repo homepage
    |-- run.py
    |-- render.yaml
    |-- requirements.txt
    |-- vrite/
    |-- ui/
    |-- ...

If you see a folder called Vrite/ instead, move the files up one level.

---

## Step 1 - Push to GitHub

Open a terminal inside the folder that contains Dockerfile:

    git init
    git add .
    git commit -m "Initial Vrite commit"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/vrite.git
    git push -u origin main

Verify: open github.com/YOUR_USERNAME/vrite and confirm Dockerfile
is visible directly on the homepage.

---

## Step 2 - Create the service on Render

1. Go to dashboard.render.com
2. Click New + then Web Service
3. Click Connect a repository
4. If GitHub is not connected yet, click Connect GitHub and authorise
5. Select your vrite repo from the list
6. Settings:
   - Name: vrite
   - Region: choose closest to you
   - Branch: main
   - Runtime: Docker  (auto-detected from Dockerfile)
   - Instance Type: Free
7. Click Create Web Service

---

## Step 3 - Set environment variables

In Render dashboard -> your service -> Environment tab -> Add env var:

    Key                  Value
    VRITE_DEVICE         cpu
    PYTHONUNBUFFERED     1

---

## Step 4 - Wait for the build

First build takes 10-20 minutes (downloads packages and clones repos).
Watch progress in the Render logs tab.

When you see:
    You can now view your Streamlit app in your browser

Your app is live at https://vrite.onrender.com (or similar URL).

---

## Free tier limits

    RAM               512 MB shared
    CPU               Shared
    Persistent disk   None (files reset on restart)
    Sleep             App sleeps after 15 min inactivity
    Voice engine      gTTS (Coqui too large without disk)
    Lip-sync          Audio-swap (no Wav2Lip weights without disk)

---

## Enable full lip-sync (needs persistent disk)

1. Render dashboard -> your service -> Disks -> Add Disk
2. Mount path: /app/models
3. Size: 10 GB
4. Cost: ~$1/month
5. Save and trigger a manual redeploy
6. The model_downloader runs automatically on next startup

---

## Keep the app awake (free tier)

Free tier apps sleep after 15 min inactivity and take ~30s to wake up.
To keep it always-on without upgrading:

1. Sign up at uptimerobot.com (free)
2. Add a new HTTP monitor pointing at:
   https://your-app.onrender.com/_stcore/health
3. Set check interval to 5 minutes

---

## Redeploy after code changes

    git add .
    git commit -m "Update"
    git push

Render redeploys automatically (autoDeploy: true in render.yaml).
