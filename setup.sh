#!/usr/bin/env bash
# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

MULTIARCH_TENSORFLOW_PYPI_REGISTRY="https://gitlab.com/api/v4/projects/37601282/packages/pypi/simple"
PIWHEELS_PYPI_REGISTRY="https://www.piwheels.org/simple"

is_command() { hash "$1" 2>/dev/null; }
apt_is_locked() { fuser /var/lib/dpkg/lock >/dev/null 2>&1; }
wait_for_apt() {
	if apt_is_locked; then
		echo "Waiting to obtain dpkg lock file..."
		while apt_is_locked; do echo .; sleep 0.5; done
	fi
}
has_piwheels_and_multiarch_tensorflow() {
    python3 -m pip config get global.extra-index-url 2>/dev/null | grep -F "$PIWHEELS_PYPI_REGISTRY" | grep -F "$MULTIARCH_TENSORFLOW_PYPI_REGISTRY"
}
install_piwheels_and_multiarch_tensorflow() {
    echo "Installing piwheels and multiarch-tensorflow indexes..."
    if ! python3 -m pip config set --global global.extra-index-url "$PIWHEELS_PYPI_REGISTRY $MULTIARCH_TENSORFLOW_PYPI_REGISTRY" > /dev/null 2>&1; then
        echo "Cannot install to system-wide configuration => installing for this site only."
        python3 -m pip config set --site global.extra-index-url "$PIWHEELS_PYPI_REGISTRY $MULTIARCH_TENSORFLOW_PYPI_REGISTRY"
    fi
}

echo '****************************************************************'
echo '** Setup for precise installation.'
echo '****************************************************************'


#############################################
set -e; cd "$(dirname "$0")" # Script Start #
#############################################

VENV=${VENV-$(pwd)/.venv}

os=$(uname -s)
if [ "$os" = "Linux" ]; then
    if is_command apt-get; then
        wait_for_apt
        if [ "$UID" == "0" ]; then
            APT_CMD="apt"
        else
            if sudo -h; then
                APT_CMD="sudo apt"
                AS_SUDO="true"
            else
                echo "ERROR: Cannot work without being root, or having sudo"
                exit 1
            fi
        fi
        $APT_CMD update 
        $APT_CMD install -yq --no-install-recommends \
            build-essential git python3-scipy cython3 libhdf5-dev python3-h5py \
            portaudio19-dev swig libpulse-dev libatlas-base-dev \
            alsa-utils
        # For pocketsphinx
        $APT_CMD install -yq --no-install-recommends swig
        $APT_CMD clean
    fi
elif [ "$os" = "Darwin" ]; then
    if is_command brew; then
        brew install portaudio
        # For pocketsphinx
        brew install swig
        export CFLAGS="$CFLAGS -I$(brew --prefix)/include"
        export LDFLAGS="$LDFLAGS -L$(brew --prefix)/lib"
    fi
fi

if [ ! -x "$VENV/bin/python" ]; then python3 -m venv "$VENV" --without-pip; fi
source "$VENV/bin/activate"
if [ ! -x "$VENV/bin/pip" ]; then
    curl https://bootstrap.pypa.io/get-pip.py | python;
else
    python3 -m pip install --upgrade pip
fi

# For debug purpose: shows compatible tags etc...
# python3 -m pip debug --verbose
# python3 -V
# python3 -c 'import platform; print(platform.uname())'
# echo OS: $os
# echo VENV: $VENV

arch="$(python -c 'import platform; print(platform.machine())')"


if ! has_piwheels_and_multiarch_tensorflow; then
    install_piwheels_and_multiarch_tensorflow
fi

echo '****************************************************************'
echo '** Installing precise-runner.'
echo '****************************************************************'
python3 -m pip install -e runner/

echo '****************************************************************'
echo '** Installing mycroft-precise.'
echo '****************************************************************'
python3 -m pip install -e .

echo '****************************************************************'
echo '** Installing pocketsphinx (accepts failure).'
echo '****************************************************************'
# Optional, for comparison
python3 -m pip install pocketsphinx || echo "Pocketsphinx installation failed (ignored)."
