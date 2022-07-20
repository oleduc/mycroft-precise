#!/usr/bin/env python3
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
from setuptools import setup

from precise import __version__, __liant_version__

from os import environ

package_name = 'mycroft-precise'
release_version = __version__,

liant_context = environ.get('PYPI_LIANT')
tag = environ.get('PYPI_TAG')
if liant_context != '' and liant_context != None:
    print('Building Liant SASU version...')
    package_name = 'liant-precise'
    release_version = __liant_version__

local_label = environ.get('PYPI_LOCAL_LABEL')
if tag != '' and tag != None:
    from packaging import version
    release_version = version.parse(tag)
    if not isinstance(release_version, version.Version):
        print(f"""
### ERROR: could not parse tag {tag} as a PEP440 version.

Failing now.
""")
        exit(1)
elif local_label != '' and local_label != None and local_label != 'main':
    print(f"Try to build with local_label: {local_label}")
    try:
        from precise import __local_version__
        release_version = release_version + '+' + local_label + '.' + __local_version__
    except ImportError:
        print(f"""
### ERROR: could not find local version.

If you want to deploy your branch package with your local label: {local_label}, please create a __local_version__ in precise/__init__.py
and maintain it.

Failing now.
""")
        exit(1)

setup(
    name=package_name,
    version=str(release_version),
    license='Apache-2.0',
    author='Matthew Scholefield',
    author_email='matthew.scholefield@mycroft.ai',
    description='Mycroft Precise Wake Word Listener',
    long_description='View more info at `the GitHub page '
                     '<https://github.com/mycroftai/mycroft-precise#mycroft-precise>`_',
    url='http://github.com/MycroftAI/mycroft-precise',
    keywords='wakeword keyword wake word listener sound',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=[
        'precise',
        'precise.scripts',
        'precise.pocketsphinx',
        'precise.pocketsphinx.scripts'
    ],
    entry_points={
        'console_scripts': [
            'precise-add-noise=precise.scripts.add_noise:main',
            'precise-collect=precise.scripts.collect:main',
            'precise-convert=precise.scripts.convert:main',
            'precise-eval=precise.scripts.eval:main',
            'precise-listen=precise.scripts.listen:main',
            'precise-listen-pocketsphinx=precise.pocketsphinx.scripts.listen:main',
            'precise-engine=precise.scripts.engine:main',
            'precise-simulate=precise.scripts.simulate:main',
            'precise-test=precise.scripts.test:main',
            'precise-graph=precise.scripts.graph:main',
            'precise-test-pocketsphinx=precise.pocketsphinx.scripts.test:main',
            'precise-train=precise.scripts.train:main',
            'precise-train-optimize=precise.scripts.train_optimize:main',
            'precise-train-sampled=precise.scripts.train_sampled:main',
            'precise-train-incremental=precise.scripts.train_incremental:main',
            'precise-train-generated=precise.scripts.train_generated:main',
            'precise-calc-threshold=precise.scripts.calc_threshold:main',
        ]
    },
    include_package_data=True,
    install_requires=[
        'numpy',
        'tensorflow==2.3.1',  # Must be on piwheels
        'sonopy',
        'pyaudio',
        'h5py',
        'wavio',
        'typing',
        'prettyparse<1.0',
        'precise-runner'
    ]
)
