import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="npaths",
    version="0.0.1",
    author="Mykhailo Tymchenko",
    author_email="mih.timchenko@gmail.com",
    description="Analytic modelling of switched N-path capacitive networks.",
    url="https://github.com/mtymchenko/npaths",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'matplotlib']
)
