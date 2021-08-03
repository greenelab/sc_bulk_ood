#!/usr/bin/env bash

cd "/content/drive/MyDrive/Colab Notebooks/checkouts/sc_bulk_ood"

echo "Ensuring ssh keys exist..."

if [ ! -f "./secrets/id_rsa" ]; then
  ssh-keygen -t ed25519 -C "natalie.davidson@cuanschutz.edu" -f "./secrets/id_rsa"
  echo "Generated key(s) $( ls ./secrets/id_rsa* )"
fi

mkdir -p "/root/.ssh/"
echo -e "Host github.com\n\tStrictHostKeyChecking no\n" > /root/.ssh/config
cp ./secrets/id_rsa "/root/.ssh/" && \
cp ./secrets/id_rsa.pub "/root/.ssh/"
echo "Copied key(s) to /root/.ssh/"
chmod go+r "/root/.ssh/id_rsa.pub"
ls -l "/root/.ssh/"
