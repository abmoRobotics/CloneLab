
from setuptools import find_namespace_packages, setup

INSTALL_REQUIRES = [
    #"skrl==1.1.0",
    "skrl",
    "torch",
    "torchvision",
    "numpy",
    "gymnasium",
    "wandb",
    "xarray",
    "PyYAML",
    "Pillow",
    "nvidia-nvimgcodec-cu12[nvjpeg,nvjpeg2k]",
    "nvidia-nvtiff-cu12",
]

setup(
    name="CloneRL",
    version="0.0.1",
    author="Anton Bjørndahl Mortensen",
    author_email="antonbm2008@gmail.com",
    keywords=["Reinforcement Learning", "Imitation Learning",
              "Offline Reinforcement Learning", "BC"],
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
    packages=find_namespace_packages(include=["CloneRL", "CloneRL.*"]),
    extras_require={
        "export": ["onnx"],
        "tensorrt": ["tensorrt"],
    },
)
