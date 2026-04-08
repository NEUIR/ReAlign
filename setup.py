from setuptools import setup, find_packages

setup(
    name='realign',
    version='1.0.0',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    python_requires='>=3.7',
)
