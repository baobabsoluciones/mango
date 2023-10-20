import setuptools

with open("README.rst") as fh:
    long_description = fh.read()

required = []
with open("requirements.txt", "r") as fh:
    required.append(fh.read().splitlines())
required = [
    el
    for el in required[0]
    if not el.startswith("#")
    and not el.startswith("pandas")
    and not el.startswith("plotly")
    and not el.startswith("beautifulsoup4")
    and not el.startswith("google-cloud-storage")
    and not el == ""
]


extra_require = {
    "gcloud": ["google-cloud-storage"],
    "data": ["pandas"],
    "plot": [
        "beautifulsoup4",
        "pandas",
        "plotly",
    ],
    "processing": [
        "pandas",
    ],
    "table": [
        "pandas",
    ],
}


setuptools.setup(
    name="mango",
    version="0.0.6",
    author="baobab soluciones",
    author_email="sistemas@baobabsoluciones.es",
    description="Library with a collection of useful classes and methods to DRY",
    long_description=long_description,
    url="https://github.com/baobabsoluciones/mango",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=required,
    extras_require=extra_require,
)
