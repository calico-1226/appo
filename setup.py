"""Setup for appo."""
from setuptools import find_packages
from setuptools import setup

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setup(
    name="appo",
    version="0.0.1",
    description=("Augmented Proximal Policy Optimization"),
    author="Juntao Dai",
    author_email="juntaodai@zju.edu.cn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    # scripts=["bin/learn"],
    install_requires=[
        'gym~=0.15.3',
        'joblib~=0.14.0',
        'mujoco_py==2.0.2.7',
        'numpy~=1.17.4',
        'xmltodict~=0.12.0',
        'tensorboardX>=2.5.1',
        'tensorboard>=2.6.0',
        'psutil>=5.8.0',
        'matplotlib>=3.4.3',
        'pandas>=1.3.3',
        'pyyaml>=5.4.1',
        # test
        "pytest>=7.0.0",
        "pre-commit>=2.17.0",
        "isort>=5.10.0",
        "black>=22.1.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="Safe Reinforcement Learning",
    
)
