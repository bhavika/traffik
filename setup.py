from setuptools import setup, find_packages

version = "0.0.1"

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="traffik",
    version=version,
    description=u"Experimenting with traffic data, mobility, graph nets",
    keywords="",
    author=u"Bhavika Tekwani",
    author_email="bhavicka@protonmail.com",
    url="https://github.com/bhavika/traffik",
    license="BSD",
    packages=find_packages(),
    exclude_package_data={'': ['data/*']},
    zip_safe=False,
    install_requires=requirements,
    extras_require={"test": ["pytest", "black"]},
)