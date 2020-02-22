#include "model_tree.hh"
#include <numeric>


template class Linear_model_tree<float>;
template class Linear_model_tree<double>;
template class Polynomial_model_tree<float>;
template class Polynomial_model_tree<double>;


template <class T>
Linear_model_tree<T>::Linear_model_tree( const std::string yaml_file_path, bool oblique ) :
                      _oblique( oblique ), _root_node( Node( 1 ) ), _node_count( 1 )
{
	YAML::Node tree_params = YAML::LoadFile( yaml_file_path );

	_build_tree_recursively( _root_node, tree_params );
}


template <class T>
void Linear_model_tree<T>::_build_tree_recursively( Node& node, const YAML::Node& tree_params )
{
	int node_id = node.id;
	if ( tree_params[node_id]["terminal"].as<bool>() )
	{
		node.params = tree_params[node_id]["model params"].as<std::vector<T>>();
	}
	else
	{
		node.terminal = false;
		node.params = tree_params[node_id]["split params"].as<std::vector<T>>();
		if ( !_oblique && node.params.size() != 2 )
			throw std::runtime_error( std::string( "Read " ) + std::to_string( (unsigned long) node.params.size() ) +
			                          std::string( " split parameters for the node " ) + std::to_string( node_id ) +
			                          std::string( " instead of the 2 required for a straight tree" ) );
		for ( int child : { 0, 1 } )
		{
			node.children[child] = node_ptr_t( _new_node() );
			_build_tree_recursively( *node.children[child], tree_params );
		}
	}
}


template <class T>
T Linear_model_tree<T>::_traverse_and_predict( const Node& node, const std::vector<T>& input, int& terminal_node_id ) const
{
	if ( ( node.terminal || _oblique ) && input.size() != node.params.size() - 1 )
		throw std::runtime_error( "The dimension of the input vector does not fit the number of parameters of the node " + std::to_string( node.id ) );

	if ( node.terminal )
	{
		terminal_node_id = node.id;
		return std::inner_product( input.begin(), input.end(), node.params.begin(), node.params.back() );
	}

	bool side;
	if ( _oblique )
		side = std::inner_product( input.begin(), input.end(), node.params.begin(), node.params.back() ) >= 0;
	else
		side = input[node.params[0]] >= node.params[1];

	return _traverse_and_predict( *node.children[ side ? 0 : 1 ], input, terminal_node_id );
}


template <class T>
T Linear_model_tree<T>::predict( const std::vector<T>& input ) const
{
	int _;
	return predict( input, _ );
}


template <class T>
T Linear_model_tree<T>::predict( const std::vector<T>& input, int& terminal_node_id ) const
{
	return _traverse_and_predict( _root_node, input, terminal_node_id );
}




template <class T>
Polynomial_model_tree<T>::Polynomial_model_tree( const std::string yaml_file_path, bool oblique, unsigned int degree, bool interaction_only ) :
                          Linear_model_tree<T>( yaml_file_path, oblique ), _degree( degree ), _interaction_only( interaction_only ) {}


template <class T>
T Polynomial_model_tree<T>::predict( const std::vector<T>& input, int& terminal_node_id ) const
{
	// Build the polynomial features:

	std::vector<T> features = input;
	std::vector<T> prev_chunk = input;
	std::vector<size_t> indices( input.size() );
	std::iota( indices.begin(), indices.end(), 0 );

	for ( int d = 1 ; d < _degree ; ++d )
	{
		// Create a new chunk of features for the degree d:
		std::vector<T> new_chunk;
		// Multiply each component with the products from the previous lower degree:
		for ( size_t i = 0 ; i < input.size() - ( _interaction_only ? d : 0 ) ; ++i )
		{
			// Store the index where to start multiplying with the current component at the next degree up:
			size_t next_index = new_chunk.size();
			for ( auto coef_it = prev_chunk.begin() + indices[i + ( _interaction_only ? 1 : 0 )] ; coef_it != prev_chunk.end() ; ++coef_it )
			{
				new_chunk.push_back( input[i]**coef_it );
			}
			indices[i] = next_index;
		}
		// Extend the feature vector with the new chunk of features:
		features.reserve( features.size() + std::distance( new_chunk.begin(), new_chunk.end() ) );
		features.insert( features.end(), new_chunk.begin(), new_chunk.end() );

		prev_chunk = new_chunk;
	}

	return this->_traverse_and_predict( this->_root_node, features, terminal_node_id );
}
