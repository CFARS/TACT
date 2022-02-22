from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as rf:
    install_requires = [m for m in rf.read().split("\n") if m]


setup(
    name='TACT',
    version='0.1.1',
    description='CFARS Site Suitability TACT Tool',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=['CFARS', 'Remote Sensing'],
    url='https://github.com/CFARS/site_suitability_tool',
    author='CFARS Remote Sensing Subgroup: [names]',
    author_email='',
    keywords='CFARS TACT remote sensing lidar sodar',
    packages=[],
    install_requires=install_requires,
    python_requires='>=3',
    zip_safe=True,
    include_package_data=True
)