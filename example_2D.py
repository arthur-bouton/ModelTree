#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from model_tree import Model_tree
import sys


######################
# REFERENCE FONCTION #
######################

# Combination of two sigmoid functions:
F = lambda x, y: max( 1/( 1 + np.exp( x - y + 5 ) ), 0.3/( 1 + np.exp( x ) ) )

x_limit = ( -10, 10 )
y_limit = ( -10, 10 )


display_resolution = 100

x_scale = np.linspace( *x_limit, display_resolution )
y_scale = np.linspace( *y_limit, display_resolution )

X, Y = np.meshgrid( x_scale, y_scale )

# Reference values to be plotted:
Zr = np.zeros( ( display_resolution, display_resolution ) )
for i, x in enumerate( x_scale ) :
	for j, y in enumerate( y_scale ) :
		Zr[j][i] = F( x, y )



################
# TRAINING SET #
################

sample_resolution = 50

data = []
for x in np.linspace( *x_limit, sample_resolution ) :
	for y in np.linspace( *y_limit, sample_resolution ) :
		data.append( [ x, y, F( x, y ) ] )
data = np.array( data )
X_data = data[:,:2]
y_data = data[:,2]



##############
# MODEL TREE #
##############

#model_tree = Model_tree( oblique=False, max_depth=5, node_min_samples=10, loss_tol=0.001, model='linear' )
model_tree = Model_tree( oblique=True, max_depth=3, node_min_samples=10, loss_tol=0.001, model='linear' )

model_tree.fit( X_data, y_data, verbose=1 )
#model_tree.save_tree_params( 'mt_params' )
#model_tree.load_tree_params( 'mt_params' )
model_tree.diagram( feature_names=[ 'x', 'y' ], float_format='{:.2f}' )

# Predictions to be plotted:
Zp = np.zeros_like( Zr )
for i, x in enumerate( x_scale ) :
	for j, y in enumerate( y_scale ) :
		Zp[j][i] = model_tree.predict( np.array([ x, y ]) )



#########
# PLOTS #
#########

azim = -20 ; elev = 25
zmin = 0 ; zmax = 1

left = 0
bottom = 0 
width = 0.95
height = 1

ls = LightSource( azdeg=320, altdeg=45 )


fig1 = plt.figure( 'Reference 2D', figsize=(8,8) )
ax = plt.axes( [ left, bottom, width, height ], projection='3d' )
rgb = ls.shade( Zr, plt.cm.coolwarm )
ax.plot_surface( X, Y, Zr, rstride=1, cstride=1, facecolors=rgb )
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_zlabel( 'z' )
ax.view_init( elev, azim )
ax.set_zlim3d( zmin, zmax )


fig2 = plt.figure( 'Prediction 2D', figsize=(8,8) )
ax = plt.axes( [ left, bottom, width, height ], projection='3d' )
rgb = ls.shade( Zp, plt.cm.coolwarm )
ax.plot_surface( X, Y, Zp, rstride=1, cstride=1, facecolors=rgb )
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_zlabel( 'z' )
ax.view_init( elev, azim )
ax.set_zlim3d( zmin, zmax )


if len( sys.argv ) > 1 and sys.argv[-1] == 'w' :
	fig1.savefig( 'pics/Reference_2D.png' )
	#fig2.savefig( 'pics/Prediction_2D_straight.png' )
	fig2.savefig( 'pics/Prediction_2D_oblique.png' )

	#model_tree.diagram( filename='pics/Diagram_example_2D_straight.png', feature_names=[ 'x', 'y' ], float_format='{:.2f}' )
	model_tree.diagram( filename='pics/Diagram_example_2D_oblique.png', feature_names=[ 'x', 'y' ], float_format='{:.2f}' )

plt.show()
