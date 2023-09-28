import os.path
import setuptools

repository_dir = os.path.dirname(__file__)

with open(os.path.join(repository_dir, "requirements.txt")) as fh:
    requirements = [line for line in fh.readlines()]

setuptools.setup(
    name="ssc_ris",
    version="1.0.0",
    author="Francisco Eiras",
    author_email="francisco.girbal@gmail.com",
    license="MIT",
    python_requires=">=3.7",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7"
    ],
    dependency_links=requirements,
    include_package_data=True,
)