import setuptools


requirements = ['torch', 'gym', 'atari-py', 'numpy', 'opencv-python']


setuptools.setup(
    name="drl",
    author="Tuan-Dat Le",
    author_email="letuandat2110@gmail.com",
    license='GPL-v3',
    description="PyTorch reinforcement learning framework",
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
