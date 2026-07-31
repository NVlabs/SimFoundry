# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages


setup(
    name="simfoundry",
    packages=[
        package for package in find_packages() if package.startswith("simfoundry")
    ],
    install_requires=[
    ],
    # configs/ holds data, not code, so it is not a package; ship its contents explicitly.
    package_data={"simfoundry": ["configs/*.yaml", "configs/modality/*.json"]},
    include_package_data=True,
    python_requires='>=3.10',
    description="SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation",
    author=(
        "Nadun Ranawaka*, Josiah Wong*, Wei-Lin Pai, Wei-Teng Chu, Tianyuan Dai, "
        "Masoud Moghani, Hang Yin, Yunfan Jiang, Wesley Durbano^, Brandon Huynh^, "
        "Yu Fang, Danfei Xu, Ruohan Zhang, Li Fei-Fei, Linxi Fan, Bowen Wen, "
        "Ajay Mandlekar†, Yuke Zhu†"
    ),
    maintainer="NVIDIA CORPORATION & AFFILIATES",
    url="https://github.com/NVlabs/SimFoundry",
    author_email="nranawakaara@nvidia.com, nadun.ranawaka@gatech.edu, jdwong@alumni.stanford.edu", 
    version="0.1.0",
)
