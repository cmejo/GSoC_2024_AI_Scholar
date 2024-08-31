#### `setup.py`

This script is for packaging the project.

```python
from setuptools import setup, find_packages

setup(
    name='rag_chatbot',
    version='0.2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'langchain',
        'transformers',
        'llama-cpp-python',
        'pypdf',
        'faiss-cpu',
    ],
    entry_points={
        'console_scripts': [
            'rag-chatbot=src.chatbot:main',
        ],
    },
)
