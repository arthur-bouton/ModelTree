#!/usr/bin/env python
from setuptools import setup

setup(
	name='modeltree',
	version='1.1',
	description='Oblique/straight model tree for regression',
	license='GPL-3.0',
	author='Arthur Bouton',
	author_email='arthur.bouton@gadz.org',
	url='https://github.com/arthur-bouton/ModelTree',
	py_modules=[ 'model_tree' ],
	install_requires=[
		'numpy>=1.16.2',
		'cma>=2.7.0',
		'PyYAML>=5.1.2',
		'scikit-learn>=0.21.3',
		'tqdm>=4.31.1',
		'graphviz>=0.16'
	]
)
