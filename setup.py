#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]


setup(
    author="A Student",
    author_email='student@example.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="IIT Elevate OSNA project",
    entry_points={
        'console_scripts': [
            'detector=detector.cli:main',
            'stats=detector.cli:stats',
            'train=detector.cli:train',
            'web=detector.cli:web',
            'predict=detector.cli:predict'
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='detector',
    name='detector',
    packages=find_packages(),
    py_modules = [],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/sheepmanchen/Twitter-Bots-Detector',
    version='0.1.0',
    zip_safe=False,
)
