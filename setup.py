from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='anneal',
      version='0.1',
      description='rnA cNvs iNferEnce And cLustering',
      url='https://github.com/Militeee/anneal',
      author='Salvatore Milite',
      author_email='militesalvatore@gmail.com',
      license='GPL-3.0',
      packages=['anneal'],
      install_requires=[
            'matplotlib',
            'pandas',
            'pyro-ppl'
      ],
      include_package_data=True,long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
      keywords='scRNA scDNA RNA CNV CNA Cancer Copy-number Bioinformatics',

      zip_safe=False)

