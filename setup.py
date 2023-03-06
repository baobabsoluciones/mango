import setuptools

with open("README.rst") as fh:
    long_description = fh.read()

required = []
# with open("requirements.txt", "r") as fh:
#     required.append(fh.read().splitlines())

extra_require = {
    "arcgis": ["fastjsonschema"],
    "email": [],
    "gcloud": ["google-cloud-storage"],
    "config": [],
    "data": ["pandas"],
    "images": [
        "cvlib",
        "dlib",
        "imutils",
        "opencv_python",
        "tensorflow",
    ],
    "logging": [],
    "model": ["numpy"],
    "plot": [
        "beautifulsoup4",
        "pandas",
        "plotly",
    ],
    "processing": [
        "numpy",
        "pandas",
        "xlsxwriter",
    ],
    "shared": ["fastjsonschema"],
}


setuptools.setup(
    name="mango",
    version="0.0.1",
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
