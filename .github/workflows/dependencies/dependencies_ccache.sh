#!/usr/bin/env bash

if [[ $# -eq 2 ]]; then
  CVER=$1
else
  CVER=4.8
fi

wget https://github.com/ccache/ccache/releases/download/v${CVER}/ccache-${CVER}-linux-x86_64.tar.xz
tar xvf ccache-${CVER}-linux-x86_64.tar.xz
sudo mv -f ccache-${CVER}-linux-x86_64/ccache /usr/local/bin/
sudo rm -rf ccache-${CVER}-linux-x86_64
sudo rm -f ccache-${CVER}-linux-x86_64.tar.xz
