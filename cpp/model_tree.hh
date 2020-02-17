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
		Node() : terminal( true ) { id = ++_node_count; }

		int id;
		bool terminal;
		std::vector<T> params;
		std::shared_ptr<Node> children[2];
	} Node;

	typedef std::shared_ptr<Node> node_ptr_t;


	Linear_model_tree( std::string yaml_file_path, bool oblique = false );

	T predict( std::vector<T>& input );


	protected:

	void _build_tree_recursively( Node& node, YAML::Node& tree_params );
	T _traverse_and_predict( Node& node, std::vector<T>& input );

	bool _oblique;
	Node _root_node;
	static int _node_count;
};


// Declaration of an alias following the C++11 standard:
template <class T>
using ptr_t = std::shared_ptr<Linear_model_tree<T>>;


#endif
