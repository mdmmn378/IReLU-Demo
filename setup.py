from setuptools import find_packages  # or find_namespace_packages
from setuptools import setup

meta = {}
with open("./src/rnn/version.py", encoding="utf-8") as f:
    exec(f.read(), meta)

setup(
    name="rnn",
    version=meta["__version__"],
    author="Mamun",
    packages=find_packages(where="src", include=["rnn"]),
    package_dir={"": "src"},
    install_requires=["torch <= 1.13.0"],
)
