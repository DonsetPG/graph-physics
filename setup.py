from setuptools import setup

setup(
    name='graph-physics',
    version='0.0.1',
    entry_points = {
        'console_scripts': ['grph=graphphysics.train:main']
    }
)
