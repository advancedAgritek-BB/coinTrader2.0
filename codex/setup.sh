#!/bin/bash
set -e

# Update apt and install basic system packages if needed
apt-get update -y
apt-get install -y --no-install-recommends git curl

# Upgrade pip and install python dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
