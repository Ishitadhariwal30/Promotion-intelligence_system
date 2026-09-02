# ============================================================
# Urban Company Promotion Intelligence Platform
# ============================================================
#
# PURPOSE
#   Package the app and everything it needs to run into one image,
#   so it behaves identically on your laptop, on a colleague's Mac,
#   and on a cloud host.
#
# BUILD    docker build -t urban-company-app .
# RUN      docker run -p 8501:8501 urban-company-app
# OPEN     http://localhost:8501
#
# TEST THE IMAGE, not just the code:
#   docker run --rm urban-company-app python tests/smoke_test.py
#
# NOT NEEDED FOR STREAMLIT COMMUNITY CLOUD - that reads
# requirements.txt and builds its own environment. This is for Azure
# Container Apps, Cloud Run, Render, or handing someone a single
# command that works.
# ============================================================


# ------------------------------------------------------------
# 1. Base image
# ------------------------------------------------------------
#
# 3.12 matches the Python the app was verified on (3.12.13). A
# different minor version can resolve different package builds,
# which is the whole class of problem this file exists to remove.
#
# `-slim` is Debian with the extras stripped - about 150 MB against
# roughly 1 GB for the full image. Smaller means faster pulls and
# less surface area. The cost is that some system libraries are
# absent, which matters immediately - see the next step.
FROM python:3.12-slim


# ------------------------------------------------------------
# 2. System libraries
# ------------------------------------------------------------
#
# LightGBM is a compiled C++ library and links against libgomp,
# the GNU OpenMP runtime. The slim image does not ship it.
#
# Leave this out and the build succeeds, the image starts, and the
# app dies on first import with:
#
#   OSError: libgomp.so.1: cannot open shared object file
#
# That error names a file you never mentioned, in a language you
# did not write. This one line is the fix, and it is the single
# most common reason a LightGBM container fails.
#
# --no-install-recommends skips optional extras. Deleting the apt
# lists in the SAME instruction matters: each instruction becomes a
# layer, so cleaning up in a later one leaves the files in the
# image anyway.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------
# 3. Working directory
# ------------------------------------------------------------
WORKDIR /app


# ------------------------------------------------------------
# 4. Dependencies, before application code
# ------------------------------------------------------------
#
# The ordering here is deliberate and is the main thing that makes
# rebuilds fast.
#
# Docker caches each instruction and reuses the cache until an input
# changes - after which every later step reruns. Dependencies change
# rarely; your code changes constantly. Copying requirements.txt on
# its own means editing a page does NOT reinstall lightgbm.
#
# Copy everything first instead and every edit costs a full reinstall.
#
# --no-cache-dir stops pip keeping its download cache in the image;
# it is never used again and only adds size.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ------------------------------------------------------------
# 5. Application code and data
# ------------------------------------------------------------
#
# .dockerignore decides what this actually copies - notably NOT
# .venv, which is 473 MB of Windows binaries.
#
# sample_data/ IS copied. The app reads its tables and model from
# there and has no other source; notebook 17 exported it precisely
# so this could run with no Databricks connection.
COPY . .


# ------------------------------------------------------------
# 6. Run as a non-root user
# ------------------------------------------------------------
#
# Containers run as root unless told otherwise. If someone finds a
# way to execute code through the app, root inside the container is
# a far better position to attack from than an unprivileged user.
#
# The app only ever reads its data, so it needs no write access.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser


# ------------------------------------------------------------
# 7. Network
# ------------------------------------------------------------
#
# EXPOSE documents the port. It does not publish it - that is the
# -p flag at run time. It is here so anyone reading this file, and
# some hosting platforms, know which port to wire up.
EXPOSE 8501


# ------------------------------------------------------------
# 8. Health check
# ------------------------------------------------------------
#
# Streamlit serves /_stcore/health once it is genuinely ready.
# Without this, an orchestrator sees "process is running" and sends
# traffic to an app still loading its model.
#
# start-period gives it 20s to load the model before failures count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"


# ------------------------------------------------------------
# 9. Start
# ------------------------------------------------------------
#
# Two flags are not optional:
#
#   --server.address=0.0.0.0
#       Streamlit binds to localhost by default. Inside a container
#       "localhost" means the container itself, so the port maps to
#       nothing and you get a blank page with no error anywhere.
#       This is the second most common containerised-Streamlit bug.
#
#   --server.headless=true
#       Stops it trying to open a browser. There is no browser here.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
