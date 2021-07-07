import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imgo",
    version="2.5.3",
    author="Elby Data",
    author_email="info@elbydata.com",
    description="Image data processing and augmentation tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elbydata/imgo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
