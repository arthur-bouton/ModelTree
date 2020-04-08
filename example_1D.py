#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from model_tree import Model_tree
import sys


n_samples = 300

x = np.linspace( 0, 10, n_samples )

y = np.sin( x[ x < 2*np.pi ] )
y = np.append( y, x[( x >= 2*np.pi )&( x < 8 )] - 2*np.pi )
y = np.append( y, -2*( x[( x >= 8 )&( x < 9 )] - 8 ) + 8 - 2*np.pi )
y = np.append( y, -1*( x[ x >= 9 ] - 9 ) + 6 - 2*np.pi )
y += np.random.randn( n_samples )*0.05



model_tree = Model_tree( oblique=False, max_depth=2, node_min_samples=10, loss_tol=0.01, model='polynomial', search_grid=10 )
model_tree.fit( x, y, verbose=1 )

x_test = np.linspace( 0, 10, 500 )
y_pred = model_tree.predict( x_test[:,np.newaxis] )



fig = plt.figure( 'Example 1D', figsize=( 10, 3.5 ) )
plt.scatter( x, y, s=8 )
plt.plot( x_test, y_pred, c='r', lw=2 )
plt.gca().set_aspect( 'equal', adjustable='box' )
plt.subplots_adjust( left=0.05, right=0.95, top=1, bottom=0 )


if len( sys.argv ) > 1 and sys.argv[-1] == 'w' :
	fig.savefig( 'pics/Example_1D.png' )

plt.show()
