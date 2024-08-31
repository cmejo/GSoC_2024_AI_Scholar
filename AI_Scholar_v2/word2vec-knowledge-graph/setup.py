#### `setup.py`

For packaging the project.

```python
from setuptools import setup, find_packages

setup(
    name='word2vec_knowledge_graph',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gensim',
        'nltk',
        'networkx',
        'matplotlib',
        'PyPDF2',
    ],
    entry_points={
        'console_scripts': [
            'knowledge-graph=src.knowledge_graph:main',
        ],
    },
)
