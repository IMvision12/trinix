from setuptools import setup, find_packages

setup(
    name="flashlm",
    version="0.1.0",
    description="Fast PyTorch attention layers with Flash Attention and Triton kernel support",
    author="ME :)",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "flash-attn>=2.0.0",
        "triton>=2.0.0",
        "numpy",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)