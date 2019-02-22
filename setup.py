from setuptools import setup
import os

def subpackages(root_dir):
    packages = [
        root_dir + '.' + dir_name
        for dir_name in os.listdir(root_dir)
        if dir_name != '__init__.py'
    ]
    return packages

setup(
    name='bmitoolbox',
    version='0.0.0',
    packages=['bmitoolbox']+subpackages('bmitoolbox'),
    install_requires=['scipy', 'numpy', 'pandas', 'h5py'],
    url='',
    license='',
    author='Yusuke ARAI',
    author_email='',
    description=''
)
