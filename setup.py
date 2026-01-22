import subprocess
import sys

packages=[
    "requests",
    "setuptools",
    "wfdb",
    "numpy",
    "matplotlib",
    "scipy",
    "neurokit2",
    "pandas<2.0",
    "yellowbrick",
    "ydata-profiling",
    "scikit-learn"
]
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in packages:
    install(pkg)