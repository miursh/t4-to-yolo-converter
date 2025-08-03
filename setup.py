from setuptools import setup, find_packages

setup(
    name="t4_to_yolo_converter",
    version="0.1.0",
    description="T4 format to YOLO format converter",
    author="Shunsuke Miura",
    packages=find_packages(),
    install_requires=[
        "pyyaml"
    ],
    entry_points={
        "console_scripts": [
            "t4-to-yolo = t4_to_yolo_converter.main:main"
        ]
    },
    python_requires=">=3.7",
)
