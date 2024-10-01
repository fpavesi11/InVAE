from setuptools import setup, find_packages

__version__ = '0.1.0'

setup(
    name='VAENAS',
    version=__version__,

    url='https://github.com/fpavesi11/NASVAE',
    author='Federico Pavesi',
    author_email='f.pavesi11@campus.unimib.it',

    packages=find_packages(include=['vaenas', 
                                    'vaenas.*',
                                    'vaenas.decoders',
                                    'vaenas.flowVAE',
                                    'vaenas.IAF',
                                    'vaenas.VanillaVAE']),
)