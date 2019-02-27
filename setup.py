from setuptools import setup

setup(
    packages = ['hdfe'],

    install_requires = [
        'numpy', 'pandas', 'scipy'
    ],

    name = 'hdfe',
    version = '0.0.1',
    description = 'Tools for working with panel data and regressions with fixed effects',
    long_description = '',

    url = 'https://github.com/esantorella/hdfe/',
    author = 'Elizabeth Santorella',
    author_email = 'elizabeth.santorella@gmail.com',
    license = 'MIT',
    platforms = [''],
    classifiers = []
)
