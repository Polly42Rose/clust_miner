import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Polly42Rose",
    version="0.0.1",
    author="Polina Tarantsova",
    author_email="pdtarantsova@edu.hse.ru",
    description="Website for constructing process models using clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Polly42Rose/clust_miner",
    project_urls={
        "Bug Tracker": "https://github.com/Polly42Rose/clust_miner/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.6.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": ""},
    packages=setuptools.find_packages(where=""),
    python_requires=">=3.6",
)