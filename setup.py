from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='congas_old',
      version='0.0.2',
      description='Copy Number genotyping from single cell RNA sequencing',
      url='https://github.com/Militeee/congas',
      author='Salvatore Milite',
      author_email='militesalvatore@gmail.com',
      license='GPL-3.0',
      packages=['congas', 'congas.models'],
      install_requires=[
            'matplotlib>=3.1',
            'pandas>=1.0',
            'pyro-ppl>=1.5',
            'numpy>=1.18',
            'scikit-learn',

      ],
      include_package_data=True,
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
      keywords='scRNA scDNA RNA CNV CNA Cancer Copy-number Bioinformatics',
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

