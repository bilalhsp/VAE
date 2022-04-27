import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VAE",
    version="0.0.1",
    author="Bilal Ahmed",
    author_email="ahmedb@purdue.edu",
    description="Variational Autocoders in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bilalhsp/",
    packages=setuptools.find_packages(),
    package_data={
        'VAE': [
            'config.yaml'
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'pandas', 'torchvision'
    ],
)
