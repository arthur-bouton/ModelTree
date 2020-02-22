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


	Linear_model_tree( const std::string yaml_file_path, bool oblique = false );

	T predict( const std::vector<T>& input ) const;
	virtual T predict( const std::vector<T>& input, int& terminal_node_id ) const;


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
using mt_ptr_t = std::shared_ptr<Linear_model_tree<T>>;


template <class T>
class Polynomial_model_tree : public Linear_model_tree<T>
{
	public:

	typedef std::shared_ptr<Polynomial_model_tree> ptr_t;

	Polynomial_model_tree( const std::string yaml_file_path, bool oblique = false, unsigned int degree = 2, bool interaction_only = false );

	using Linear_model_tree<T>::predict;
	T predict( const std::vector<T>& input, int& terminal_node_id ) const;


	protected:

	unsigned int _degree;
	bool _interaction_only;
};


#endif
