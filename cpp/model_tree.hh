#ifndef MODEL_TREE_HH
#define MODEL_TREE_HH 

#include <yaml-cpp/yaml.h>


template <class T>
class Linear_model_tree
{
	public:

	typedef std::shared_ptr<Linear_model_tree> ptr_t;

	typedef struct Node
	{
		Node( int id ) : id( id ), terminal( true ) {}

		int id;
		bool terminal;
		std::vector<T> params;
		std::shared_ptr<Node> children[2];
	} Node;

	typedef std::shared_ptr<Node> node_ptr_t;


	Linear_model_tree( const std::string yaml_file_path, const bool oblique = false );

	T predict( const std::vector<T>& input ) const;
	T predict( const std::vector<T>& input, int& terminal_node_id ) const;


	protected:

	inline Node* _new_node() { return new Node( ++_node_count ); }

	void _build_tree_recursively( Node& node, const YAML::Node& tree_params );
	T _traverse_and_predict( const Node& node, const std::vector<T>& input, int& terminal_node_id ) const;

	bool _oblique;
	Node _root_node;
	int _node_count;
};


// Declaration of an alias following the C++11 standard:
template <class T>
using lmt_ptr_t = std::shared_ptr<Linear_model_tree<T>>;


#endif
