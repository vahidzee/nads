from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nads',
    packages=find_packages(exclude=['examples', 'docs']),  # add other exclusions in future
    version='0.0.4',
    license='MIT',
    description='Neural Anisotropy Directions Toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Vahid Zehtab',
    author_email='vahid98zee@gmail.com',
    url='https://github.com/vahidzee/nads',
    keywords=[
        'artificial intelligence',
        'inductive bias',
        'deep learning'
    ],
    install_requires=[
        'torch>=1.9',
        'matplotlib',
        'tqdm'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
