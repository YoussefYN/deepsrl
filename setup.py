from setuptools import setup

setup(
    name='deepsrl',
    version='0.0.0',
    description='DeepSRL models for semantic roles analysis.',
    url='git@github.com:YoussefYN/deepsrl.git',
    author='Youssef Youssry',
    author_email='y.ibrahim@innopolis.university',
    license='unlicense',
    packages=['deepsrl', 'deepsrl.neural_srl', 'deepsrl.neural_srl.shared', 'deepsrl.neural_srl.theano'],
    install_requires=[
        'numpy',
        'theano',
        'protobuf',
        'nltk'
    ],
    zip_safe=False
)