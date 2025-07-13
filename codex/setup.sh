#!/bin/bash
set -e

# Update apt and install basic system packages if needed
apt-get update -y
apt-get install -y --no-install-recommends git curl python3 python3-pip

# Upgrade pip and install python dependencies
python3 -m pip install --upgrade pip
# requirements.txt includes numpy and other packages needed for running the tests
python3 -m pip install -r requirements.txt
