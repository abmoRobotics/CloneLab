
import os

from setuptools import setup

INSTALL_REQUIRES = [
    "skrl==0.10.0",
    "torch",
    "numpy",
    "gymnasium",
    "wandb",
]

setup(
    name="CloneRL",
    version="0.0.1",
    author="Anton Bjørndahl Mortensen",
    author_email="antonbm2008@gmail.com",
    keywords=["Reinforcement Learning", "Imitation Learning", "Offline Reinforcement Learning", "BC"],
    description="A library for offline reinforcement learning and imitation learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/abmoRobotics/CloneRL",
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    license="MIT",
    zip_safe=False,
    packages=["CloneRL"]
)