from setuptools import find_packages, setup


setup(
    name="learning",
    version="0.0.1",
    packages=find_packages(where="."),
    keywords="duckietown, environment, agent, reinforcement",
    include_package_data=True
)
