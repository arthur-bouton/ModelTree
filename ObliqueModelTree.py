#!/usr/bin/env python
import numpy as np
from sklearn.metrics import mean_squared_error
import cma
import warnings


class Linear_Regression :

	def __init__( self ) :
		from sklearn.linear_model import LinearRegression

		self.model = LinearRegression()

	def fit( self, X, y ) :
		self.model.fit( X, y )

	def predict( self, X ) :
		return self.model.predict( X )

	def get_coefs( self ) :
		return np.append( self.model.coef_, self.model.intercept_ )


class Polynomial_Regression :

	def __init__( self, degree=2 ) :
		from sklearn.preprocessing import PolynomialFeatures
		from sklearn.linear_model import LinearRegression

		self.preprocessing = PolynomialFeatures( degree, include_bias=False )
		self.model = LinearRegression()

	def fit( self, X, y ) :
		X_poly = self.preprocessing.fit_transform( X ) 
		self.model.fit( X_poly, y )

	def predict( self, X ) :
		X_poly = self.preprocessing.fit_transform( X ) 
		return self.model.predict( X_poly )

	def get_coefs( self ) :
		return np.append( self.model.coef_, self.model.intercept_ )


def CMA_Search( X, cost_function, verbose=False, indentation=0 ) :

	data_center = np.mean( X, 0 )
	data_maxvar = np.max( np.var( X, 0 ) )

	xi = np.random.randn( X.shape[1] )
	xi = np.append( xi, xi.dot( data_center ) )

	#es = cma.CMAEvolutionStrategy( xi, data_maxvar, { 'verbose': 0, 'verb_log': 0, 'verb_disp': 0 } )
	es = cma.CMAEvolutionStrategy( xi, data_maxvar, { 'verbose': 0, 'verb_log': 0, 'verb_disp': 0, 'tolconditioncov': False } )
	with warnings.catch_warnings() :
		warnings.simplefilter( "ignore" )
		es.optimize( cost_function )

	if verbose :
		print( '%saCMA-ES results -> n iter: %i -- n eval: %i -- termination status: %s' % ( ' '*indentation, es.result.iterations, es.result.evaluations, es.stop() ) )
		#es.result_pretty()

	coef_max = max( abs( es.result.xbest ) )*( -1 if es.result.xbest[0] < 0 else 1 )
	opti_coefs = es.result.xbest/coef_max

	return opti_coefs


class Oblique_Model_Tree :

	def __init__( self, max_depth=5, min_samples_leaf=1, model='linear', loss_tol=None, margin_coef=0.01, split_search='cma-es' ) :

		if model == 'linear' :
			self.model = Linear_Regression
		elif model == 'second_order' :
			self.model = lambda : Polynomial_Regression( 2 )
		else :
			self.model = model

		if split_search == 'cma-es' :
			self.split_search = CMA_Search
		else :
			self.split_search = split_search

		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.loss_tol = loss_tol
		self.margin_coef = margin_coef

		self._root_node = None


	def _get_split_distribution( self, X, split_coefs ) :
		if X.ndim == 1 : X = X[np.newaxis,:]
		assert len( split_coefs ) == X.shape[1] + 1

		boundary = split_coefs[0]*X[:,0] - split_coefs[-1]
		for i in range( 1, X.shape[1] ) :
			boundary += split_coefs[i]*X[:,i]

		return boundary >= 0


	def _divide_and_fit( self, X, y, split_coefs, margin_coef=0.01 ) :
		assert len( y ) == X.shape[0]

		distribution = self._get_split_distribution( X, split_coefs )

		X_1 = X[distribution]
		y_1 = y[distribution]

		X_2 = X[~distribution]
		y_2 = y[~distribution]

		# If there is not enough samples on one side of the split, return the squared distance from the center of the samples:
		if len( y_1 ) < self.min_samples_leaf or len( y_2 ) < self.min_samples_leaf :
			data_center = np.mean( X, 0 )
			nosplit_loss = ( ( data_center.dot( split_coefs[:-1] ) - split_coefs[-1] )/np.linalg.norm( split_coefs[:-1] ) )**2
			results = { 'success': False,
						'split_loss': nosplit_loss,
						'margin_penalty': 0 }
			#print( 'OUT no-split loss:', nosplit_loss )
			return results

		model_1 = self.model()
		model_1.fit( X_1, y_1 )
		y_pred_1 = model_1.predict( X_1 )
		loss_1 = mean_squared_error( y_1, y_pred_1 )

		model_2 = self.model()
		model_2.fit( X_2, y_2 )
		y_pred_2 = model_2.predict( X_2 )
		loss_2 = mean_squared_error( y_2, y_pred_2 )

		split_loss = ( len( y_1 )*loss_1 + len( y_2 )*loss_2 )/len( y )

		# Add the maximization of the margin:
		margin_distance = min( abs( X.dot( split_coefs[:-1] ) - split_coefs[-1] ) )/np.linalg.norm( split_coefs[:-1] )
		margin_penalty = -margin_coef*margin_distance

		results = { 'success': True,
		            'split_loss': split_loss,
					'margin_penalty': margin_penalty,
					'models': ( model_1, model_2 ),
					'model_losses': ( loss_1, loss_2 ),
					'X': ( X_1, X_2 ),
					'y': ( y_1, y_2 ) }
		return results
	

	def _create_node( self, depth=0, model=None ) :
		node = { 'depth': depth,
				 'model': model,
				 'terminal': True }
		return node

	
	def _split_recursively( self, node, X, y, verbose=1, loss=None ) :

		if verbose :
			print( '  %s\u21B3Depth %i: n samples: %i%s'
			% ( '    '*node['depth'], node['depth'], len( y ), ( ' -- Model loss: %g' % loss ) if loss is not None else '' ) )

		if node['depth'] >= self.max_depth or len( y ) < 2*self.min_samples_leaf :
			if loss is None :
				node['model'] = self.model()
				node['model'].fit( X, y )
				y_pred = node['model'].predict( X )
				loss = mean_squared_error( y, y_pred )
			if verbose :
				print( '  %s*Terminal (%s)%s'
				% ( '    '*node['depth'], ( 'max depth' if node['depth'] >= self.max_depth else 'sample limit' ), self._print_model_coefs( node['model'] ) ) )
			return

		def cost_function( v ) :
			results = self._divide_and_fit( X, y, v )
			return results['split_loss'] + results['margin_penalty']

		# Search for the optimal split:
		split_coefs = self.split_search( X, cost_function, verbose > 1, 3 + 4*node['depth'] )

		# Get the data and models from the optimal split:
		results = self._divide_and_fit( X, y, split_coefs )

		if results['success'] :
			if verbose :
				print( '  %s Split loss: %g -- Margin penalty: %g%s'
				% ( '    '*node['depth'], results['split_loss'], results['margin_penalty'], ( ' -- Split parameters: %s' % split_coefs ) if verbose > 2 else '' ) )

			node['terminal'] = False
			node['split_coefs'] = split_coefs

			node['children'] = []
			for child in range( 2 ) :
				node['children'].append( self._create_node( node['depth'] + 1, results['models'][child] ) )
				if self.loss_tol is None or results['model_losses'][child] > self.loss_tol :
					self._split_recursively( node['children'][child], results['X'][child], results['y'][child], verbose, results['model_losses'][child] )

				elif verbose :
					print( '  %s\u21B3Depth %i: n samples: %i -- Model loss: %g'
					% ( '    '*( node['depth'] + 1 ), node['depth'] + 1, len( results['y'][child] ), results['model_losses'][child] ) )
					print( '  %s*Terminal (loss tol)%s'
					% ( '    '*( node['depth'] + 1 ), self._print_model_coefs( results['models'][child] ) ) )

		elif verbose :
			print( '  %s Could not find a proper split! -- Split loss: %g%s'
			% ( '    '*node['depth'], results['split_loss'], self._print_model_coefs( node['model'] ) ) )


	#def _print_terminal_node( self, depth, n_samples, loss=None, model=None ) :
		#return ''


	def _print_model_coefs( self, model ) :
		try :
			return ( ' -- Model parameters: %s' % model.get_coefs() ) if self.verbose > 2 else ''
		except AttributeError :
			return ''


	def fit( self, X, y, verbose=1 ) :

		if verbose :
			print( 'Oblique Model Tree -- max depth: %i -- min samples leaf: %i -- loss tol: %s -- margin coef: %g'
			% ( self.max_depth, self.min_samples_leaf, ( '%g' % self.loss_tol ) if self.loss_tol is not None else 'None', self.margin_coef ) )

		self._root_node = self._create_node()
		self._split_recursively( self._root_node, X, y, verbose )

		if verbose :
			y_pred = self.predict( X )
			loss = mean_squared_error( y, y_pred )
			print( 'Final loss: %g' % loss )


	def _traverse_nodes_and_predict( self, node, x, verbose=0 ) :

		if node['terminal'] :
			if x.ndim == 1 : x = x[np.newaxis,:]
			return node['model'].predict( x )

		side = 0 if self._get_split_distribution( x, node['split_coefs'] ) else 1
		return self._traverse_nodes_and_predict( node['children'][side], x, verbose )


	def predict( self, X, verbose=0 ) :
		if X.ndim == 1 :
			return self._traverse_nodes_and_predict( self._root_node, X[np.newaxis,:], verbose ).item()
		else :
			return np.hstack([ self._traverse_nodes_and_predict( self._root_node, x, verbose ) for x in X ])


	def _traverse_nodes_and_get_model_coefs( self, node ):
		if node['terminal'] :
			return [ node["model"].get_coefs() ]
		else :
			coef_list = []
			for child in range( 2 ) :
				coef_list.extend( self._traverse_nodes_and_get_model_coefs( node["children"][child] ) )
			return coef_list


	def get_model_coefs( self ) :
		if self._root_node is not None :
			return self._traverse_nodes_and_get_model_coefs( self._root_node )
		else :
			return None


