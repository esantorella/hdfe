from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    packages=["hdfe"],
    install_requires=["numpy", "pandas>=0.25.0", "scipy"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="hdfe",
    version="0.0.4",
    description="Econometric tools for working with panel data and fixed effects",
    url="https://github.com/esantorella/hdfe/",
    author="Elizabeth Santorella",
    author_email="elizabeth.santorella@gmail.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
