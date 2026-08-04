#!/usr/bin/env bash

set -eu -o pipefail

if [[ $# -eq 1 ]]; then
  CVER=$1
else
  CVER=4.13.6
fi

wget https://github.com/ccache/ccache/releases/download/v${CVER}/ccache-${CVER}-linux-x86_64-glibc.tar.xz
tar xvf ccache-${CVER}-linux-x86_64-glibc.tar.xz
sudo mv -f ccache-${CVER}-linux-x86_64-glibc/ccache /usr/local/bin/
sudo rm -rf ccache-${CVER}-linux-x86_64-glibc
sudo rm -f ccache-${CVER}-linux-x86_64-glibc.tar.xz
