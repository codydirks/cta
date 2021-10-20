import setuptools
from cta import __version__

setuptools.setup(
    name='cta',
    version=__version__,
    author='Cody Dirks',
    packages=setuptools.find_packages(where='cta'),
    install_requires=[
        'pandas',
        'jupytext',
        'torchcast @ git+https://github.com/strongio/torchcast.git@develop#egg=torchcast',
        'mock'
    ]
)
