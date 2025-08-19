#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import time
from setuptools import find_packages, setup

version_file = 'fga/version.py'
default_version = '0.1.0'

def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def _minimal_ext_cmd(cmd):
    # construct minimal environment (for stable git calls)
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
    return out


def get_git_hash():
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except Exception:
        sha = 'unknown'
    return sha


def get_hash_short():
    if os.path.exists('.git'):
        return get_git_hash()[:7]
    return 'unknown'


def read_short_version():
    if os.path.exists('VERSION'):
        with open('VERSION', 'r', encoding='utf-8') as f:
            return f.read().strip()
    return default_version


def write_version_py():
    short_version = read_short_version()
    sha = get_hash_short()

    def _to_tuple(v):
        parts = []
        for x in v.split('.'):
            parts.append(x if not x.isdigit() else int(x))
        return tuple(parts)

    version_info_tuple = _to_tuple(short_version)

    content = f"""# GENERATED VERSION FILE
                # TIME: {time.asctime()}
                __version__ = '{short_version}'
                __gitsha__ = '{sha}'
                version_info = {version_info_tuple}
                """
    os.makedirs(os.path.dirname(version_file), exist_ok=True)
    with open(version_file, 'w', encoding='utf-8') as f:
        f.write(content)


def get_version():
    if not os.path.exists(version_file):
        write_version_py()
    scope = {}
    with open(version_file, 'r', encoding='utf-8') as f:
        code = compile(f.read(), version_file, 'exec')
        exec(code, scope)
    return scope['__version__']


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    req_path = os.path.join(here, filename)
    if not os.path.exists(req_path):
        return []
    with open(req_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines()]
    requires = [ln for ln in lines if ln and not ln.startswith('#')]
    return requires


if __name__ == '__main__':
    write_version_py()
    setup(
        name='fga-sr',
        version=get_version(),
        description='FGA-SR: Fourier-Guided Attention Upsampling for Image Super-Resolution',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='Daejune Choi, Youchan No, Jinhyung Lee, Duksu Kim',
        author_email='eowns02@gmail.com',
        keywords='computer vision, pytorch, image restoration, super-resolution, fourier, attention',
        url='https://github.com/HPC-Lab-KOREATECH/FGA-SR',
        project_urls={
            'Paper': 'https://arxiv.org/abs/2508.10616',
            'Source': 'https://github.com/HPC-Lab-KOREATECH/FGA-SR',
            'Issues': 'https://github.com/HPC-Lab-KOREATECH/FGA-SR/issues',
        },
        include_package_data=True,
        packages=find_packages(
            exclude=(
                'options',
                'datasets',
                'experiments',
                'results',
                'weights',
                'assets',
                'scripts',
                'data',
                'notebooks',
                'tb_logger',
                'wandb',
                'docs',
            )
        ),
        classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        license='Apache-2.0',
        install_requires=get_requirements(),
        python_requires='>=3.8',
        zip_safe=False,
    )
