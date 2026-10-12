# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Editable install:  pip install -e .   (the CUDA extensions JIT-build for sm_70 on first import;
# the bundled cutlass headers in ./fa2_src must stay alongside the package for the prefill kernel).
from setuptools import setup, find_packages

setup(
    name="fa2_sm70",
    version="0.3.0",
    description=("FlashAttention-2 for NVIDIA V100 / Volta (sm_70): exact prefill + differentiable "
                 "backward (training) + a decode stack (split-KV, int8/int4 KV, paged, stateful), "
                 "with varlen, dropout, ALiBi, soft-capping, and a flash-attn-compatible API."),
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="xormal (Alexander Romanoff)",
    license="MIT",
    url="https://github.com/xormal/V100-SM_70-Flash-Attn-2-v1",
    packages=find_packages(include=["fa2_sm70", "fa2_sm70.*"]),
    package_data={"fa2_sm70": ["csrc/*.cu"]},
    include_package_data=True,
    install_requires=["torch>=2.1"],   # needs torch.nn.attention.sdpa_kernel + a CUDA (cu12x) build
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
