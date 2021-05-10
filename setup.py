from setuptools import setup, find_packages

setup(
    name='tf-x-transformers',
    packages=find_packages(),
    version='0.0.1',
    license='MIT',
    description='TF-X-Transformers - TF2.x',
    author='Junn Yu',
    author_email='573009727@qq.com',
    url='https://github.com/junnyu/tf-x-transformers',
    keywords=[
        'artificial intelligence', 'attention mechanism', 'transformers'
        'tensorflow 2.x'
    ],
    install_requires=[
        'tensorflow>=2.2', 'einops>=0.3', 'fastcore', 'tf_fast_api'
    ],
)