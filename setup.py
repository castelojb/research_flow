from setuptools import setup, find_packages

setup(
    name="research_flow",
    version="0.1.0",
    author="João Castelo Branco",
    author_email="seuemail@dominio.com",
    description="Descrição do seu pacote",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/castelojb/research_flow.git",
    packages=find_packages(),
    install_requires=[
        "gloe>=0.5.9",
        "pydantic>=2.7.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)