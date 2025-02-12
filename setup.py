from setuptools import setup, find_packages

setup(
    name='synthesizer',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',

    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 
                      'seaborn', 'scikit-learn', 'pubchempy', 'tqdm', 'joblib', 'GPy', 'GPyOpt'],
    
    author='Leo Luber',
    author_email= 'l.luber@campus.lmu.de',
    description='An optimizer for perovskite nanocrystal synthesis',
    long_description=open('README.md').read(),
    keywords='perovskite nanocrystals synthesis optimization',
    url= 'https://github.com/leoluber/synthesizer',
)