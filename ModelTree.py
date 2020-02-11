#!/usr/bin/env python
import numpy as np
from sklearn.metrics import mean_squared_error
import cma
import warnings
import yaml


class Linear_regression :

	def __init__( self, L1_reg=None ) :
		self._L1_reg = L1_reg

		if L1_reg is None :
			from sklearn.linear_model import LinearRegression
			self.model = LinearRegression()
		else :
			from sklearn.linear_model import Lasso
			self.model = Lasso( L1_reg )

	def fit( self, X, y ) :
		self.model.fit( X, y )

	def predict( self, X ) :
		return self.model.predict( X )

	def get_params( self ) :
		return np.append( self.model.coef_, self.model.intercept_ ).tolist()

	def set_params( self, params ) :
		self.model.coef_ = np.array( params[:-1] )
		self.model.intercept_ = params[-1]
		self.model.n_iter_ = 0

	def __str__( self ) :
		return 'Linear regression%s' % ( ' with a L1 regularization coefficient of %g' % self._L1_reg if self._L1_reg is not None else '' )


class Polynomial_regression( Linear_regression ) :

	def __init__( self, degree=2, L1_reg=None ) :
		self._degree = degree

		from sklearn.preprocessing import PolynomialFeatures
		self.preprocessing = PolynomialFeatures( degree, include_bias=False )

		Linear_regression.__init__( self, L1_reg )

	def fit( self, X, y ) :
		X_poly = self.preprocessing.fit_transform( X ) 
		self.model.fit( X_poly, y )

	def predict( self, X ) :
		X_poly = self.preprocessing.fit_transform( X ) 
		return self.model.predict( X_poly )

	def __str__( self ) :
		return 'Polynomial regression of degree %i%s' % ( self._degree, ( ' with a L1 regularization coefficient of %g' % self._L1_reg ) if self._L1_reg is not None else '' )


def CMA_search( X, cost_function, verbose=False, indentation=0 ) :

	data_center = np.mean( X, 0 )
	data_maxvar = np.max( np.var( X, 0 ) )

	x0 = np.random.randn( X.shape[1] )
	x0 = np.append( x0, x0.dot( data_center ) )

	#es = cma.CMAEvolutionStrategy( x0, data_maxvar, { 'verbose': 0, 'verb_log': 0, 'verb_disp': 0 } )
	es = cma.CMAEvolutionStrategy( x0, data_maxvar, { 'verbose': 0, 'verb_log': 0, 'verb_disp': 0, 'tolconditioncov': False } )
	with warnings.catch_warnings() :
		warnings.simplefilter( "ignore" )
		es.optimize( cost_function )

	if verbose :
		print( '%saCMA-ES results -> %i iterations, %i evaluations, termination status: %s' % ( ' '*indentation, es.result.iterations, es.result.evaluations, es.stop() ) )
		#es.result_pretty()

	return es.result.xbest


class Model_tree :

	def __init__( self, oblique=True, max_depth=5, min_samples_leaf=1, model='linear', loss_tol=None, split_search='cma-es', margin_coef=0.01, **model_options ) :

		if model == 'linear' :
			self.model = lambda : Linear_regression( **model_options )
		elif model == 'polynomial' :
			self.model = lambda : Polynomial_regression( **model_options )
		else :
			self.model = model

		if split_search == 'cma-es' :
			self.split_search = CMA_search
		else :
			self.split_search = split_search

		self.oblique = oblique
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.loss_tol = loss_tol
		self.margin_coef = margin_coef

		self._root_node = None


	def _get_split_distribution( self, X, split_params ) :
		if X.ndim == 1 : X = X[np.newaxis,:]

		if self.oblique :
			assert len( split_params ) == X.shape[1] + 1

			boundary = split_params[0]*X[:,0] - split_params[-1]
			for i in range( 1, X.shape[1] ) :
				boundary += split_params[i]*X[:,i]

		else :
			assert len( split_params ) == 2 and isinstance( split_params[0], int )

			boundary = X[:,split_params[0]] - split_params[1]

		return boundary >= 0


	def _divide_and_fit( self, X, y, split_params, margin_coef=0.01 ) :
		assert len( y ) == X.shape[0]

		distribution = self._get_split_distribution( X, split_params )

		X_1 = X[distribution]
		y_1 = y[distribution]

		X_2 = X[~distribution]
		y_2 = y[~distribution]

		# If there is not enough samples on one side of the split, return the squared distance from the center of the samples:
		if len( y_1 ) < self.min_samples_leaf or len( y_2 ) < self.min_samples_leaf :
			if self.oblique :
				nosplit_loss = ( ( np.mean( X, 0 ).dot( split_params[:-1] ) - split_params[-1] )/np.linalg.norm( split_params[:-1] ) )**2
			else :
				nosplit_loss = ( np.mean( X[:,split_params[0]] ) - split_params[1] )**2
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
		if self.oblique :
			margin_distance = min( abs( X.dot( split_params[:-1] ) - split_params[-1] ) )/np.linalg.norm( split_params[:-1] )
		else :
			#margin_distance = min( abs( X[:,split_params[0]] - split_params[1] ) )
			margin_distance = 0
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
		if depth == 0 :
			self._node_count = 1
		else :
			self._node_count += 1

		node = { 'depth': depth,
				 'id': self._node_count,
				 'model': model,
				 'terminal': True }
		return node


	def _split_recursively( self, node, X, y, verbose=1, loss=None ) :

		# If the maximum depth is reached or there is not enough samples left:
		if node['depth'] >= self.max_depth or len( y ) < 2*self.min_samples_leaf :
			if loss is None :
				node['model'] = self.model()
				node['model'].fit( X, y )
				y_pred = node['model'].predict( X )
				loss = mean_squared_error( y, y_pred )
			if verbose :
				self._print_terminal_node( node['id'], node['depth'], 'max depth' if node['depth'] >= self.max_depth else 'samp limit', len( y ), loss )
			return

		if self.oblique :
			# Declaration of the cost function for the split:
			def cost_function( p ) :
				results = self._divide_and_fit( X, y, p )
				return results['split_loss'] + results['margin_penalty']

			# Search for the optimal split:
			split_params = self.split_search( X, cost_function, verbose > 1, 2 + 4*node['depth'] )

			# Normalize by the highest coefficient and impose the first coefficient to be positive:
			coef_max = max( np.abs( split_params ) )*( -1 if split_params[0] < 0 else 1 )
			split_params = ( np.array( split_params )/coef_max ).tolist()

		else :
			for feature in range( X.shape[1] ) :
				# Identify all possible thresholds in the middle of each successive pair of samples:
				feature_values = np.unique( X[:,feature] ).tolist()
				threshold_list = [ ( feature_values[i+1] + feature_values[i] )/2 for i in range( self.min_samples_leaf - 1, len( feature_values ) - self.min_samples_leaf ) ]

				# Record the best split seen:
				for threshold in threshold_list :
					results = self._divide_and_fit( X, y, ( feature, threshold ) )
					if 'best_split_loss' not in locals() or results['split_loss'] < best_split_loss :
						best_split_loss = results['split_loss']
						split_params = [ feature, threshold ]

		# Get the data and models from the optimal split:
		results = self._divide_and_fit( X, y, split_params )

		if results['success'] :
			if verbose :
				# Proportion of samples in each split:
				prop1 = len( results['y'][0] )/len( y )*100
				prop2 = len( results['y'][1] )/len( y )*100
				print( '  %s\u21B3Split node (%i) at depth %i (%.f%%/%.f%% of %i samples, split loss: %g%s)'
				% ( '    '*node['depth'], node['id'], node['depth'], prop1, prop2, len( y ), results['split_loss'],
				    ', margin penalty: %g' % results['margin_penalty'] if self.oblique else ', threshold on feature [%i]: %g' % tuple( split_params ) ) )

			node['terminal'] = False
			node['split_params'] = split_params
			node['model'] = None # Free the model resources

			node['children'] = []
			for child in range( 2 ) :
				node['children'].append( self._create_node( node['depth'] + 1, results['models'][child] ) )

				# If the loss is not below the tolerance:
				if self.loss_tol is None or results['model_losses'][child] > self.loss_tol :
					self._split_recursively( node['children'][child], results['X'][child], results['y'][child], verbose, results['model_losses'][child] )

				elif verbose :
					self._print_terminal_node( node['children'][child]['id'], node['depth'] + 1, 'loss tol', len( results['y'][child] ), results['model_losses'][child] )

		elif verbose :
			print( '  %s\u21B3/!\ Node (%i) at depth %i: Could not find a suitable split (%i samples, split loss: %g)'
			% ( '    '*node['depth'], node['id'], node['depth'], len( y ), results['split_loss'] ) )


	def _print_terminal_node( self, node_id, depth, termination_str, n_samples, loss ) :
		print( '  %s\u21B3Terminal node (%i) at depth %i (%s)%s %*i samples, model loss: %g'
		% ( '    '*depth, node_id, depth, termination_str, ' '*abs( 10 - len( termination_str ) ), self._nsamples_len, n_samples, loss ) )


	def fit( self, X, y, verbose=1 ) :
		
		if verbose :
			print( self.__str__() )
			print( 'Model: %s' % str( self.model() ) )
			self._nsamples_len = len( str( len( y ) ) )

		self._root_node = self._create_node()
		self._split_recursively( self._root_node, X, y, verbose )

		if verbose :
			y_pred = self.predict( X )
			loss = mean_squared_error( y, y_pred )
			print( 'Final loss: %g, number of nodes: %i, number of non-zero parameters: %i, total number of parameters: %i'
			% ( loss, self._node_count, *self.get_number_of_params() ) )


	def _traverse_nodes_and_predict( self, node, x, return_node_id=False ) :

		if node['terminal'] :
			if x.ndim == 1 : x = x[np.newaxis,:]
			if return_node_id :
				return node['model'].predict( x ), node['id']
			else :
				return node['model'].predict( x )

		side = 0 if self._get_split_distribution( x, node['split_params'] ) else 1
		return self._traverse_nodes_and_predict( node['children'][side], x, return_node_id )


	def predict( self, X, return_node_id=False ) :
		if self._root_node is None :
			raise RuntimeError( 'The tree has not been built yet' )

		if X.ndim == 1 :
			ret = self._traverse_nodes_and_predict( self._root_node, X[np.newaxis,:], return_node_id )
			if return_node_id :
				return ret[0].item(), ret[1]
			else :
				return ret.item()
		else :
			ret_list = [ self._traverse_nodes_and_predict( self._root_node, x, return_node_id ) for x in X ]
			if return_node_id :
				return np.hstack( [ ret[0] for ret in ret_list ] ), [ ret[1] for ret in ret_list ]
			else :
				return np.hstack( ret_list )


	def _traverse_nodes_and_collect_params( self, node, parent_id=None, tree_params={} ) :
		node_info = 'Node %i%s' % ( node['id'], ( ' at depth %i, child of node %i' % ( node['depth'], parent_id ) ) if parent_id is not None else ', root' )
		tree_params[node['id']] = { 'info': node_info }
		if node['terminal'] :
			tree_params[node['id']]['terminal'] = True
			tree_params[node['id']]['model params'] = node['model'].get_params()
			return tree_params
		else :
			tree_params[node['id']]['terminal'] = False
			tree_params[node['id']]['split params'] = node['split_params']

			for child in range( 2 ) :
				self._traverse_nodes_and_collect_params( node['children'][child], node['id'], tree_params )

			return tree_params


	def get_tree_params( self ) :
		'''Return all node parameters describing the tree as a dictionary'''
		if self._root_node is None :
			return None

		return self._traverse_nodes_and_collect_params( self._root_node )


	def _traverse_nodes_and_count_params( self, node ) :
		if node['terminal'] :
			params = node['model'].get_params()
			nonzero = sum( x != 0 for x in params )
			total = len( params )
			return nonzero, total
		else :
			params = node['split_params']
			if self.oblique :
				nonzero = sum( x != 0 for x in params ) - 1
				total = len( params ) - 1
			else :
				nonzero = 1
				total = 1

			for child in range( 2 ) :
				n, t = self._traverse_nodes_and_count_params( node['children'][child] )
				nonzero += n
				total += t

			return nonzero, total


	def get_number_of_params( self ) :
		'''Return a tuple with first the number of non-zero parameters and second the total number of parameters'''
		if self._root_node is None :
			return 0

		return self._traverse_nodes_and_count_params( self._root_node )


	def _set_params_recursively( self, node, tree_params ) :
		if tree_params[node['id']]['terminal'] :
			node['model'] = self.model()
			node['model'].set_params( tree_params[node['id']]['model params'] )
		else :
			node['terminal'] = False
			node['split_params'] = tree_params[node['id']]['split params']
			node['children'] = {}
			for child in range( 2 ) :
				node['children'][child] = self._create_node( node['depth'] + 1 )
				self._set_params_recursively( node['children'][child], tree_params )


	def set_tree_params( self, tree_params ) :
		'''Build the tree according to a dictionary with the same structure as returned by get_tree_params()'''
		self._root_node = self._create_node()
		self._set_params_recursively( self._root_node, tree_params )


	def save_tree_params( self, filename ) :
		with open( filename + '.yaml', 'w') as f :
			f.write( '# %s\n' % self.__str__() )
			f.write( '# Model: %s\n' % str( self.model() ) )
			yaml.dump( self.get_tree_params(), f, sort_keys=False )


	def load_tree_params( self, filename ) :
		with open( filename + '.yaml', 'r') as f :
			self.set_tree_params( yaml.load( f, Loader=yaml.FullLoader ) )


	def __str__( self ) :
		return '%s Model Tree (max depth: %i, min samples leaf: %i, loss tol: %s, margin coef: %g)' %\
		( 'Oblique' if self.oblique else 'Straight', self.max_depth, self.min_samples_leaf, '%g' % self.loss_tol if self.loss_tol is not None else 'None', self.margin_coef )


