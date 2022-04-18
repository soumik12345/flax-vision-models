import os
import re
import setuptools


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r'^__version__ = ["\']([^"\']*)["\']', version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    dependencies = f.read().splitlines()


setuptools.setup(
    name="flax-vision-models",
    version=find_version('flax_models', '__init__.py'),
    author="Soumik Rakshit",
    author_email="19soumik.rakshit96@gmail.com",
    description="A repository of Deep Learning models in Flax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/soumik12345/flax-vision-models",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "flax_models"},
    packages=setuptools.find_packages(exclude=("notebooks/")),
    python_requires=">=3.7.12,<3.11",
    install_requires=dependencies,
)
