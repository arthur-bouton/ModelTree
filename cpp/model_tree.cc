#include "model_tree.hh"
#include <numeric>


template class Linear_model_tree<float>;
template class Linear_model_tree<double>;


template <class T>
int Linear_model_tree<T>::_node_count;


template <class T>
Linear_model_tree<T>::Linear_model_tree( const std::string yaml_file_path, const bool oblique ) : _oblique( oblique )
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
			throw std::runtime_error( std::string( "Read " ) + std::to_string( (int) node.params.size() ) +
			                          std::string( " split parameters for the node " ) + std::to_string( node_id ) +
			                          std::string( " instead of the 2 required for a straight tree" ) );
		for ( int child : { 0, 1 } )
		{
			node.children[child] = node_ptr_t( new Node );
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
	return _traverse_and_predict( _root_node, input, _ );
}


template <class T>
T Linear_model_tree<T>::predict( const std::vector<T>& input, int& terminal_node_id ) const
{
	return _traverse_and_predict( _root_node, input, terminal_node_id );
}
