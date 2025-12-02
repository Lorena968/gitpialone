#!/bin/bash
set -e

mkdir -p /opt/sipa-ind
cp -r * /opt/sipa-ind/

pip install -r /opt/sipa-ind/requirements.txt

cp sipa-ind.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable sipa-ind.service
systemctl start sipa-ind.service

