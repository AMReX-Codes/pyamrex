#!/usr/bin/env bash

set -eu -o pipefail

if [[ $# -eq 1 ]]; then
  CVER=$1
else
  CVER=4.13.6
fi

# ccache publishes Linux release binaries named after `uname -m`, which covers
# the x64 and the arm64 GitHub-hosted runners.
CARCH=$(uname -m)
case "${CARCH}" in
  x86_64|aarch64) ;;
  *) echo "No ccache ${CVER} release binary for ${CARCH}" >&2; exit 1 ;;
esac
CNAME=ccache-${CVER}-linux-${CARCH}-glibc

wget https://github.com/ccache/ccache/releases/download/v${CVER}/${CNAME}.tar.xz
tar xvf ${CNAME}.tar.xz
sudo mv -f ${CNAME}/ccache /usr/local/bin/
sudo rm -rf ${CNAME}
sudo rm -f ${CNAME}.tar.xz
