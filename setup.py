from setuptools import setup, find_packages
import os

with open("README.md", encoding="utf-8") as readme_file:
    README = readme_file.read()

setup(
    name='voiceldm',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[str(r) for r in open(os.path.join(os.path.dirname(__file__), "requirements.txt"))],
    author='Yeonghyeon Lee',
    author_email='glory20h@kaist.ac.kr',
    description="VoiceLDM: Text-to-Speech with Environmental Context",
    long_description=README,
    long_description_content_type="text/markdown",
)
