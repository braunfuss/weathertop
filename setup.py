#!/usr/bin/env python

from setuptools import setup
from setuptools.command.install import install


class CustomInstallCommand(install):
    def run(self):
        install.run(self)


setup(
    cmdclass={
        'install': CustomInstallCommand,
    },

    name='weathertop',

    description='A python based image processing tool for surface displacements.',

    version='0.1',

    author='Andreas Steinberg',

    author_email='andreas.steinberg@ifg.uni-kiel.de',

    packages=[
        'weathertop',
        'weathertop.apps',
        'weathertop.process',
        'weathertop.plotting',
    ],
    python_requires='>=3.5',
    entry_points={
        'console_scripts': [
            'weathertop = weathertop.apps.weathertop:main',
            'weathertop_clients = weathertop.apps.weathertop_clients:main',

        ]
    },
    package_dir={'weathertop': 'src'},

    data_files=[],

    license='GPLv3',

    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        ],

    keywords=[
        'seismology, waveform analysis, earthquake modelling, geophysics,'],
    )
