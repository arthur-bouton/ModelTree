#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from model_tree import Model_tree, Polynomial_regression
import sys


# Definition of the training set:

n_samples = 300

x = np.linspace( 0, 10, n_samples )

y = np.sin( x[ x < 2*np.pi ] )
y = np.append( y, x[( x >= 2*np.pi )&( x < 8 )] - 2*np.pi )
y = np.append( y, -2*( x[( x >= 8 )&( x < 9 )] - 8 ) + 8 - 2*np.pi )
y = np.append( y, -1*( x[ x >= 9 ] - 9 ) + 6 - 2*np.pi )
y += np.random.randn( n_samples )*0.05


# Definition of the test set:

x_test = np.linspace( 0, 10, 500 )


# Declaration and training of a model tree:

model_tree = Model_tree( oblique=False, max_depth=3, node_min_samples=10, model='polynomial', degree=2, search_grid=10 )
model_tree.fit( x, y, verbose=1 )
y_pred_0 = model_tree.predict( x_test[:,np.newaxis] )


# Fitting of polynomial regressions for comparison:

poly_regr_1 = Polynomial_regression( degree=4 )
poly_regr_1.fit( x.reshape(-1, 1), y )
y_pred_1 = poly_regr_1.predict( x_test[:,np.newaxis] )

poly_regr_2 = Polynomial_regression( degree=8 )
poly_regr_2.fit( x.reshape(-1, 1), y )
y_pred_2 = poly_regr_2.predict( x_test[:,np.newaxis] )


# Plots:

fig = plt.figure( 'Example 1D', figsize=( 10, 4 ) )
plt.scatter( x, y, s=12 )
plt.plot( x_test, y_pred_0, c='r', lw=2 )
plt.plot( x_test, y_pred_1, c='c', lw=2 )
plt.plot( x_test, y_pred_2, c='y', lw=2 )
plt.legend( [ 'Model tree with polynomial regressions of degree 2', 'Polynomial regression of degree 4', 'Polynomial regression of degree 8' ], loc=3 )
plt.gca().set_aspect( 'equal', adjustable='box' )
plt.subplots_adjust( left=0.04, right=0.96, top=1, bottom=0.06 )


if len( sys.argv ) > 1 and sys.argv[-1] == 'w' :
	fig.savefig( 'pics/Example_1D.png' )

plt.show()
