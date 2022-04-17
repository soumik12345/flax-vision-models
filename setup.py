import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    dependencies = f.read().splitlines()


setuptools.setup(
    name="flax-vision-models",
    version="0.0.1",
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
    package_dir={"": "flax_models"},
    packages=setuptools.find_packages(where="flax_models"),
    python_requires=">=3.8.10,<3.11",
    install_requires=dependencies,
)
