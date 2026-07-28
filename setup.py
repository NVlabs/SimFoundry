# Copyright (c) 2024 the ACDC authors (Stanford Vision and Learning Lab)
# Licensed under the Apache License, Version 2.0.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file is derived from the ACDC / digital-cousins project
# (https://github.com/cremebrule/digital-cousins). NVIDIA modifications are licensed
# under Apache-2.0; the adapted upstream portions remain subject to the terms above.

from setuptools import setup, find_packages


setup(
    name="simfoundry",
    packages=[
        package for package in find_packages() if package.startswith("digital_cousins")
    ],
    install_requires=[
    ],
    eager_resources=['*'],
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
    author_email="nadun.ranawaka@gatech.edu, jdwong@alumni.stanford.edu", 
    version="0.1.0",
)
