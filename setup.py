from setuptools import setup, find_packages


setup(
    name="digital-cousins",
    packages=[
        package for package in find_packages() if package.startswith("digital_cousins")
    ],
    install_requires=[
    ],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3.10',
    description="Automated Creation of Digital Cousins for Robust Policy Learning",
    author="Tianyuan Dai*, Josiah Wong*, Yunfan Jiang, Chen Wang, Cem Gokmen, Ruohan Zhang, Jiajun Wu, Li Fei-Fei",
    url="https://github.com/cremebrule/digital-cousins",
    author_email="tydai@stanford.edu, jdwong@stanford.edu",
    version="0.0.1",
)
