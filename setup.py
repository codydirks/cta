import setuptools
from cta import __version__

setuptools.setup(
    name='cta',
    version=__version__,
    author='Cody Dirks',
    packages=setuptools.find_packages(where='cta'),
    install_requires=[
        'pandas',
        'jupytext'
    ]
)
