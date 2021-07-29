from setuptools import setup, find_packages

setup(
    name='nads',
    packages=find_packages(exclude=['examples', 'docs']),  # add other exclusions in future
    version='0.0.1',
    license='MIT',
    description='Neural Anisotropy Directions Toolbox',
    author='Vahid Zehtab',
    author_email='vahid98zee@gmail.com',
    url='https://github.com/vahidzee/nads',
    keywords=[
        'artificial intelligence',
        'inductive bias',
        'deep learning'
    ],
    install_requires=[
        'torch>=1.6',
        'torchvision'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
