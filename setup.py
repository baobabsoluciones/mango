import setuptools

with open("README.rst") as fh:
    long_description = fh.read()

required = []
# with open("requirements.txt", "r") as fh:
#     required.append(fh.read().splitlines())

extra_require = {
    "arcgis": [],
    "email": [],
    "gcloud": ["google-cloud-storage"],
    "config": [],
    "data": ["pandas"],
    "images": [
        "cvlib==0.2.7",
        "dlib==19.24.0",
        "imutils==0.5.4",
        "keras==2.9.0",
        "opencv_python==4.6.0.66",
        "tensorflow==2.9.0",
        "sklearn",
    ],
    "logging": [],
    "model": ["numpy"],
    "plot": ["pandas", "plotly"],
    "processing": ["numpy", "pandas"],
}



setuptools.setup(
    name="mango",
    version="0.0.1a8",
    author="baobab soluciones",
    author_email="sistemas@baobabsoluciones.es",
    description="Library with a collection of useful classes and methods to DRY",
    long_description=long_description,
    url="https://github.com/baobabsoluciones/mlfunctions",
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
