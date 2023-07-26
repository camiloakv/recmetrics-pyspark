import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="recmetrics-pyspark",
    version="0.0.1",
    author="Camilo Akimushkin Valencia",
    author_email="camilo.akimushkin@gmail.com",
    description="A library of recommender systems metrics for big data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/camiloakv/recmetrics-pyspark",
    packages=["recmetrics_pyspark"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)