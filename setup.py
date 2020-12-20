import setuptools

with open('./README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(name='nntools',
                 version='0.1.0',
                 author='Riccardo Finotello',
                 author_email='riccardo.finotello@gmail.com',
                 description='Tools for deep learning and data analysis',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/thesfinox/nntools',
                 packages=setuptools.find_packages(),
                 classifiers=['Programming Language :: Python :: 3',
                              'License :: OSI Approved :: MIT License',
                              'Operating System :: OS Independent',
                              'Topic :: Scientific/Engineering :: Mathematics',
                              'Topic :: Scientific/Engineering :: Physics'
                             ],
                 install_requires=['numpy',
                                   'pandas',
                                   'tensorflow>=2',
                                   'seaborn',
                                   'matplotlib'
                                  ],
                 python_requires='>=3.6'
                ) 
