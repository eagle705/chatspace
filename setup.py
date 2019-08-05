from setuptools import find_packages, setup

with open("README.rst") as f:
    long_description = f.read()

setup(
    name="chatspace",
    version="1.0.1",
    description="spacing model for korean text with chat-style",
    author="Suin Seo, Junseong Kim",
    author_email="developers@scatterlab.co.kr",
    url="https://github.com/pingpong-ai/chatspace",
    install_requires=[
        "torch"
    ],
    packages=find_packages(exclude=['tests']),
    keywords=["spaceing", "korean", "pingpong"],
    python_requires=">=3.6",
    long_description=long_description,
    license="Apache License 2.0",
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache License 2.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Natural Language :: Korean",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
