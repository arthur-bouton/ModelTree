# Oblique/straight model tree for regression


Regression trees approximate a function by partitioning the input space and applying an independent regression model on each of the subdivisions.<br />
A **straight** tree divides the input space in two at each node by selecting a single feature and splitting at a threshold.<br />
An **oblique** tree divides the input space in two at each node with an hyperplane. This way, the partitioning can accommodate multivariate frontiers and greatly reduce the number of nodes needed.

Below is an example for which the training set is provided by the composition of two sigmoid functions. Both model trees tries to reproduce the reference function with a collection of linear regressions. The straight tree can easily model the sigmoid function that is aligned with a single feature but struggle to approximate the multivariate sigmoid function. Conversely, the oblique tree can find a more relevant segmentation which allows a shallower tree to achieve a better match.

Reference function | Straight tree | Oblique tree
:-:|:-:|:-:
![](pics/Reference_2D.png?raw=true) | ![](pics/Prediction_2D_straight.png?raw=true) | ![](pics/Prediction_2D_oblique.png?raw=true)
|| (max_depth=5) | (max_depth=3)

The source file of this example is *example_2D.py*.


Here is another example with a single-dimension input space and piecewise second-order regressions:

![](pics/Example_1D.png?raw=true "1D example")

The source file of this example is *example_1D.py*.


### Usage

The first argument when initializing a Model_tree object is a boolean specifying if the tree is oblique.

If true, the argument `split_search` let us specify the algorithm to be used in order to find the optimal split at each node. The function to pass has to take an array of data and the cost function as arguments and return the set of parameters for the best hyperplane found, as suggested by the default function *CMA_search* defined in *model_tree.py*.

By default, `split_search='cma-es'`, resulting in the use of the Covariance Matrix Adaptation Evolution Strategy implemented [here](https://github.com/CMA-ES/pycma "github.com/CMA-ES/pycma").

To search for the best split, the cost function to minimize is defined by:

<!-- $$ L_{split} = \frac{n_1 L_1 + n_2 L_2}{n_1 + n_2} - \lambda\min\frac{| X \cdot w + b |}{\| w \|} $$ -->
<p align="center">
	<img src="https://render.githubusercontent.com/render/math?math=\Large{}L_{split}=\frac{n_1L_1%2Bn_2L_2}{n_1%2Bn_2}-\lambda\min\frac{|X\cdot{}w%2Bb|}{\|w\|}">
</p>

where *L<sub>i</sub>* is the loss obtained by the regression model applied to the *n<sub>i</sub>* samples of the side *i* of the current split, while *w* and *b* are the coefficients of the hyperplane defining this split. Therefore, the second term of the cost function favors the maximization of the margin between the samples and the hyperplane. The coefficient ![lambda](https://render.githubusercontent.com/render/math?math=\lambda), by default 0.01, can be changed via the argument `margin_coef`.

Otherwise, if the tree is straight, an exhaustive research is performed for every cut possible along every feature. This process can be speed up with the argument `search_grid` which let us specify an interval number of possible thresholds to skip at a first scan pass. The second scan pass then looks for all thresholds in the best interval found previously.

Wether the tree is oblique or not, the model to use at each terminal node is specified by the argument `model`. It can be one of the strings `'linear'` or `'polynomial'` in order to use linear or polynomial regressions with L1 and L2 regularizations implemented with the [scikit-learn](https://scikit-learn.org "scikit-learn.org") library. Otherwise, a class describing any particular model can be provided as long as it implements the same methods as the classes *Linear_regression* and *Polynomial_regression* defined in *model_tree.py*.

Any argument can be passed to the model by keyword. For example, to declare an oblique polynomial model tree of degree 3 with a L2 regularization of 0.01 and a maximum depth of 5 nodes, you would write:

`tree = Model_tree( oblique=True, max_depth=5, model='polynomial', degree=3, L2=0.01 )`

The tree is then trained with the method `fit( X, y, verbose=1 )`, where X is the array of training data with the shape ( n samples, n features ) and y the targets.<br />
If X is a single dimension array, it is considered to be n samples with a single feature.<br />
If verbose = 0, no output is given.<br />
If verbose = 1, only the outputs from the tree building are displayed.<br />
If verbose = 2, outputs from the split search are provided as well.

The predictions are obtained with the method `predict( X, return_node_id=False )`, where X is an array of input data with the shape ( n samples, n features ).<br />
If X is a single dimension array, it is considered to be one sample of n features.<br />
If return_node_id is True, return a tuple containing first the list of predictions and second a list of the corresponding terminal node numbers.


### All the arguments for the initialization of a Model_tree object

`oblique=False`: each split is made according to a scalar threshold on a single feature (straight tree).<br />
`oblique=True`: splits are defined by a linear combination of all features (hyperplane in the feature space).<br />
`max_depth`: maximum depth of the tree.<br />
`node_min_samples`: minimum number of training samples to be used to constitute and train a terminal node.<br />
`model`: regression model to use at each terminal node.<br />
`loss_tol`: tolerance on the model loss at which to stop splitting.<br />
`split_search`: function used for searching the oblique split coefficients. If split_search='cma-es', Covariance Matrix Adaptation Evolution Strategy is used (oblique trees only).<br />
`margin_coef`: coefficient used to incite the maximization of the margin with the training samples (oblique trees only).<br />
`search_grid`: interval number of possible thresholds to skip for the first scan pass of a grid search (straight trees only).<br />
`**model_options`: options to be passed to the regression model of each terminal node.


### Saving and loading trained trees

The parameters of the current tree can be saved in a YAML file with the method `save_tree_params( 'filename' )` and then restored with the method `load_tree_params( 'filename' )`.<br />
The parameters can also be extracted as a dictionary with `get_tree_params()` and set manually with `set_tree_params( a_dictionary_of_parameters )`.


### Use a trained model tree in a C++ program

A linear or polynomial model tree trained in python can then be imported in a C++ program thanks to the class templates *Linear_model_tree* and *Polynomial_model_tree* defined in *cpp/model_tree.cc*. To import the parameters from a YAML file, you will need the library [yaml-cpp](https://github.com/jbeder/yaml-cpp 'github.com/jbeder/yaml-cpp') which can be installed on Debian-based systems with:

`sudo apt-get install libyaml-cpp-dev`

Below is an example showing how to use the model tree in C++:

```
#include "model_tree.hh"

main()
{
	// Declaration of an oblique model tree with second-order regressions:
	Polynomial_model_tree<double> tree( "./mt_params.yaml", true, 2 );

	// Declaration of the input (two-dimensional case):
	std::vector<double> input = { -2, 3 };
	// Integer used to store the number of the terminal node (optional):
	int node_id;

	// Inference of the model tree output:
	double output = tree.predict( input, node_id );

	printf( "Output value: %g -- Corresponding node: %d\n", output, node_id );
}
```
