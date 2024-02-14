import setuptools

setuptools.setup(
    name="dlcalc",
    version="0.1.0",
    packages=["dlcalc"],
    entry_points={
        "console_scripts": [
            "3dtraining = dlcalc.3d_training_memory:main",
        ],
    },
)