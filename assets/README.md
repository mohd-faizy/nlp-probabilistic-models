# Visual Assets Catalog

This directory houses all theoretical diagrams, notebook figures, architectural schematics, and media assets for the repository.

---

## Catalog Structure

Visual assets are divided into two primary categories:
1. **Theoretical Foundations & Figures (`fig_01` - `fig_61`)**: Standardized, sequential mathematical figures referenced within the notes and documentation in [`README.md`](../README.md).
2. **Notebook Figures & Architecture Diagrams**: Clean, descriptive functional diagrams directly referenced by the interactive Jupyter notebooks.

---

## 1. Notebook & Architectural Diagrams

| Asset Filename | Format | Concept / Diagram Description | Notebook Reference |
| :--- | :--- | :--- | :--- |
| `banner.png` | `.png` | Repository header banner for Natural Language Processing with Probabilistic Models. | [`README.md`](../README.md) |
| `autocorrect_architecture.png` | `.png` | 4-step autocorrect pipeline: identify misspelled words, generate candidates $n$ edits away, filter candidates, and calculate probabilities. | [`03_autocorrect_and_minimum_edit_distance.ipynb`](../03_autocorrect_and_minimum_edit_distance.ipynb) |
| `string_splits_diagram.png` | `.png` | Visual breakdown of string splitting into prefixes $L$ and suffixes $R$ for edit generation. | [`03_autocorrect_and_minimum_edit_distance.ipynb`](../03_autocorrect_and_minimum_edit_distance.ipynb) |
| `string_switches_diagram.png` | `.png` | Visual breakdown of adjacent character transpositions (switching). | [`03_autocorrect_and_minimum_edit_distance.ipynb`](../03_autocorrect_and_minimum_edit_distance.ipynb) |
| `list_comprehension_deletes.png` | `.png` | Python list comprehension mechanism for single-character deletion candidates. | [`03_autocorrect_and_minimum_edit_distance.ipynb`](../03_autocorrect_and_minimum_edit_distance.ipynb) |
| `generic_list_comprehension.png` | `.png` | Generic list comprehension workflow and syntax diagram. | [`03_autocorrect_and_minimum_edit_distance.ipynb`](../03_autocorrect_and_minimum_edit_distance.ipynb) |
| `edit_distance_initialization.png` | `.png` | Initialization of the dynamic programming distance matrix $D[i, 0]$ and $D[0, j]$. | [`03_autocorrect_and_minimum_edit_distance.ipynb`](../03_autocorrect_and_minimum_edit_distance.ipynb) |
| `edit_distance_example.png` | `.png` | Step-by-step worked example comparing source and target words. | [`03_autocorrect_and_minimum_edit_distance.ipynb`](../03_autocorrect_and_minimum_edit_distance.ipynb) |
| `edit_distance_dynamic_table.png` | `.png` | Populated 2D dynamic programming grid showing minimum edit distance backpointers. | [`03_autocorrect_and_minimum_edit_distance.ipynb`](../03_autocorrect_and_minimum_edit_distance.ipynb) |
| `pos_data_sources_diagram.png` | `.png` | Penn Treebank Wall Street Journal training and test data sources. | [`06_pos_tagging_and_viterbi_algorithm.ipynb`](../06_pos_tagging_and_viterbi_algorithm.ipynb) |
| `pos_tagging_diagram.png` | `.png` | Example sentence demonstrating syntactic Part-of-Speech tag annotations. | [`06_pos_tagging_and_viterbi_algorithm.ipynb`](../06_pos_tagging_and_viterbi_algorithm.ipynb) |
| `viterbi_initialization_matrix.png` | `.png` | Dynamic programming initialization of probability matrix $C$ and state matrix $D$. | [`06_pos_tagging_and_viterbi_algorithm.ipynb`](../06_pos_tagging_and_viterbi_algorithm.ipynb) |
| `viterbi_forward_pass_trellis.png` | `.png` | Viterbi forward pass recursion trellis across time steps $t = 1 \dots K$. | [`06_pos_tagging_and_viterbi_algorithm.ipynb`](../06_pos_tagging_and_viterbi_algorithm.ipynb) |
| `viterbi_backward_pass_trace.png` | `.png` | Traceback path through matrix $D$ identifying the optimal hidden state sequence. | [`06_pos_tagging_and_viterbi_algorithm.ipynb`](../06_pos_tagging_and_viterbi_algorithm.ipynb) |
| `stanford_treebank_logo.png` | `.png` | Stanford Sentiment Treebank and language modeling reference banner. | [`10_autocomplete_ngram_language_model.ipynb`](../10_autocomplete_ngram_language_model.ipynb) |
| `cbow_model_architecture.png` | `.png` | Neural architecture of the Continuous Bag-of-Words model (Input $\to$ Hidden $\to$ Output). | [`12_cbow_model_architecture.ipynb`](../12_cbow_model_architecture.ipynb), [`15_cbow_word_embeddings_practice.ipynb`](../15_cbow_word_embeddings_practice.ipynb) |
| `cbow_model_dimensions.png` | `.png` | Tensor and weight dimensions ($V \times N$, $N \times V$) for single and batch CBOW examples. | [`13_cbow_model_training.ipynb`](../13_cbow_model_training.ipynb), [`15_cbow_word_embeddings_practice.ipynb`](../15_cbow_word_embeddings_practice.ipynb) |
| `cbow_mean_vector.png` | `.png` | Aggregation of context word one-hot vectors into the average context vector $\bar{x}$. | [`16_word_embeddings_continuous_bag_of_words.ipynb`](../16_word_embeddings_continuous_bag_of_words.ipynb) |
| `cbow_word_vectors.png` | `.png` | Matrix representation of dense word vectors extracted from trained weights. | [`16_word_embeddings_continuous_bag_of_words.ipynb`](../16_word_embeddings_continuous_bag_of_words.ipynb) |

---

## 2. Theoretical Foundations & Figures (`fig_01` – `fig_61`)

All theoretical figures are formatted as high-resolution PNGs and cataloged below:

| Figure ID | Filename | Topic & Concept | Location in `README.md` |
| :--- | :--- | :--- | :--- |
| **Fig 01** | `fig_01_string_edit_operations.png` | Combining insert, delete, switch, and replace edit operations | [Building the model](../README.md#building-the-model) |
| **Fig 02** | `fig_02_candidate_filtering.png` | Filtering generated candidates against vocabulary lexicon | [Building the model](../README.md#building-the-model) |
| **Fig 03** | `fig_03_word_probabilities.png` | Calculating empirical word probability in corpus | [Building the model](../README.md#building-the-model) |
| **Fig 04** | `fig_04_minimum_edit_distance_costs.png` | Minimum edit distance operation costs (insert, delete, replace) | [Minimum edit distance](../README.md#minimum-edit-distance) |
| **Fig 05** | `fig_05_edit_distance_matrix_setup.png` | Dynamic programming matrix setup (source word vs. target word) | [Minimum edit distance algorithm](../README.md#minimum-edit-distance-algorithm) |
| **Fig 06** | `fig_06_edit_distance_recursion_formula.png` | Formulaic recurrence relation for dynamic table cell values | [Minimum edit distance algorithm](../README.md#minimum-edit-distance-algorithm) |
| **Fig 07** | `fig_07_pos_tagging_sentence_example.png` | Part-of-speech tagged sentence parsing example | [Part of Speech Tagging](../README.md#part-of-speech-tagging) |
| **Fig 08** | `fig_08_markov_chain_state_graph.png` | Directed graph of Markov chain states and transition probabilities | [Markov Chains](../README.md#markov-chains) |
| **Fig 09** | `fig_09_pos_tag_transition_graph.png` | POS tag state transitions and edge weights | [Markov Chains and POS Tags](../README.md#markov-chains-and-pos-tags) |
| **Fig 10** | `fig_10_pos_transition_matrix_probabilities.png` | Transition probability matrix $A$ dimensions and row normalization | [Markov Chains and POS Tags](../README.md#markov-chains-and-pos-tags) |
| **Fig 11** | `fig_11_hidden_markov_model_emission_matrix.png` | Hidden Markov Model emission probabilities matrix $B$ | [Hidden Markov Chains models](../README.md#hidden-markov-chains-models) |
| **Fig 12** | `fig_12_transition_matrix_tag_counts.png` | Tag pair frequency count formulation $C(t_{i-1}, t_i)$ | [The Transition Matrix](../README.md#the-transition-matrix) |
| **Fig 13** | `fig_13_transition_matrix_calculation.png` | Estimating transition probabilities from tag counts | [The Transition Matrix](../README.md#the-transition-matrix) |
| **Fig 14** | `fig_14_transition_matrix_laplace_smoothing.png` | Laplace smoothing formulation for transition matrix | [The Transition Matrix](../README.md#the-transition-matrix) |
| **Fig 15** | `fig_15_emission_matrix_cooccurrences.png` | Tag-word emission co-occurrence count formulation | [The Emission Matrix](../README.md#the-emission-matrix) |
| **Fig 16** | `fig_16_viterbi_algorithm_state_trellis.png` | Viterbi state trellis graph for optimal path decoding | [The Viterbi Algorithm](../README.md#the-viterbi-algorithm) |
| **Fig 17** | `fig_17_viterbi_matrix_initialization.png` | Initialization of probability matrix $C$ and state matrix $D$ | [Initialization](../README.md#initialization) |
| **Fig 18** | `fig_18_viterbi_forward_pass_formula.png` | Recursive equation for matrix $C$ in the forward pass | [Forward Pass](../README.md#forward-pass) |
| **Fig 19** | `fig_19_viterbi_forward_pass_backpointers.png` | Maximizing index selection for backpointer matrix $D$ | [Forward Pass](../README.md#forward-pass) |
| **Fig 20** | `fig_20_viterbi_backward_pass_log_probabilities.png` | Log-probability summation to mitigate underflow in backward pass | [Backward Pass](../README.md#backward-pass) |
| **Fig 21** | `fig_21_ngram_definition_and_examples.png` | Definition of $N$-gram word sequences | [N-Grams and Probabilities](../README.md#n-grams-and-probabilities) |
| **Fig 22** | `fig_22_unigram_probability_formula.png` | Unigram probability formula | [N-Grams and Probabilities](../README.md#n-grams-and-probabilities) |
| **Fig 23** | `fig_23_bigram_conditional_probability.png` | Bigram conditional probability formula | [N-Grams and Probabilities](../README.md#n-grams-and-probabilities) |
| **Fig 24** | `fig_24_ngram_conditional_probability.png` | General $N$-gram conditional probability equation | [N-Grams and Probabilities](../README.md#n-grams-and-probabilities) |
| **Fig 25** | `fig_25_sequence_probability_chain_rule.png` | Probability chain rule decomposition for full sentences | [Sequence Probabilities](../README.md#sequence-probabilities) |
| **Fig 26** | `fig_26_markov_assumption_bigram_approximation.png` | Bigram Markov assumption approximating sequence probabilities | [Sequence Probabilities](../README.md#sequence-probabilities) |
| **Fig 27** | `fig_27_ngram_count_matrix.png` | Constructing the $N$-gram frequency count matrix | [The N-gram Language Model](../README.md#the-n-gram-language-model) |
| **Fig 28** | `fig_28_ngram_probability_matrix.png` | Row-normalizing count matrix to conditional probabilities | [The N-gram Language Model](../README.md#the-n-gram-language-model) |
| **Fig 29** | `fig_29_ngram_language_model_probabilities.png` | Connecting probability matrix to statistical language modeling | [The N-gram Language Model](../README.md#the-n-gram-language-model) |
| **Fig 30** | `fig_30_corpus_train_val_test_split.png` | Train/validation/test corpus splitting strategies | [Language Model evaluation](../README.md#language-model-evaluation) |
| **Fig 31** | `fig_31_language_model_perplexity_metric.png` | Perplexity ($PP$) evaluation metric formula | [Language Model evaluation](../README.md#language-model-evaluation) |
| **Fig 32** | `fig_32_bigram_perplexity_formula.png` | Perplexity formula specialized for bigram models | [Language Model evaluation](../README.md#language-model-evaluation) |
| **Fig 33** | `fig_33_log_perplexity_formula.png` | Log-perplexity computation for numerical stability | [Language Model evaluation](../README.md#language-model-evaluation) |
| **Fig 34** | `fig_34_ngram_sparsity_and_zero_counts.png` | Zero-frequency sparsity in corpus distributions | [Smoothing](../README.md#smoothing) |
| **Fig 35** | `fig_35_laplacian_add_one_smoothing.png` | Laplace (add-one) probability smoothing formula | [Smoothing](../README.md#smoothing) |
| **Fig 36** | `fig_36_add_k_smoothing.png` | Add-$k$ (Lidstone) smoothing formula | [Smoothing](../README.md#smoothing) |
| **Fig 37** | `fig_37_linear_interpolation_smoothing.png` | Linear interpolation smoothing across $N$-gram orders | [Smoothing](../README.md#smoothing) |
| **Fig 38** | `fig_38_word_integer_indexing.png` | Integer vocabulary representation | [Basic Word Representations](../README.md#basic-word-representations) |
| **Fig 39** | `fig_39_one_hot_vector_representation.png` | Sparse one-hot vector encoding | [Basic Word Representations](../README.md#basic-word-representations) |
| **Fig 40** | `fig_40_dense_word_embeddings.png` | Low-dimensional dense word vector embeddings | [Word Embeddings](../README.md#word-embeddings) |
| **Fig 41** | `fig_41_word_embedding_learning_pipeline.png` | End-to-end word embedding generation pipeline | [Word Embeddings](../README.md#word-embeddings) |
| **Fig 42** | `fig_42_cbow_context_and_center_words.png` | Continuous Bag-of-Words context and center target words | [Continuous Bag-of-Words Model](../README.md#continuous-bag-of-words-model) |
| **Fig 43** | `fig_43_cbow_sliding_window_extraction.png` | Sliding window extraction of training pairs | [Continuous Bag-of-Words Model](../README.md#continuous-bag-of-words-model) |
| **Fig 44** | `fig_44_cbow_one_hot_training_pairs.png` | Final prepared one-hot training vectors for CBOW | [Transforming Words into Vectors](../README.md#transforming-words-into-vectors) |
| **Fig 45** | `fig_45_cbow_neural_network_architecture.png` | Shallow neural network architecture for CBOW | [Architecture of the CBOW Model](../README.md#architecture-of-the-cbow-model) |
| **Fig 46** | `fig_46_cbow_layer_dimensions_single_input.png` | CBOW layer dimensions for a single training input | [CBOW Model Dimensions](../README.md#cbow-model-dimensions) |
| **Fig 47** | `fig_47_cbow_layer_dimensions_batch.png` | CBOW layer dimensions for mini-batch processing | [CBOW Model Dimensions](../README.md#cbow-model-dimensions) |
| **Fig 48** | `fig_48_cbow_relu_activation.png` | Rectified Linear Unit (ReLU) activation function | [Activation Functions](../README.md#activation-functions) |
| **Fig 49** | `fig_49_cbow_softmax_activation.png` | Softmax probability normalization function | [Activation Functions](../README.md#activation-functions) |
| **Fig 50** | `fig_50_cbow_cross_entropy_loss.png` | Cross-entropy loss for a single training observation | [Cost Functions](../README.md#cost-functions) |
| **Fig 51** | `fig_51_cbow_batch_cost_function.png` | Batch cross-entropy cost function over $m$ examples | [Cost Functions](../README.md#cost-functions) |
| **Fig 52** | `fig_52_cbow_forward_propagation_equations.png` | CBOW forward propagation mathematical equations | [Forward Propagation](../README.md#forward-propagation) |
| **Fig 53** | `fig_53_cbow_forward_propagation_matrix_forms.png` | Forward propagation matrix equations for hidden layer $h$ and output $\hat{y}$ | [Forward Propagation](../README.md#forward-propagation) |
| **Fig 54** | `fig_54_cbow_backpropagation_gradients.png` | Backpropagation partial derivatives ($\partial J / \partial W_1$, $\partial J / \partial W_2$) | [Backpropagation and Gradient Descent](../README.md#backpropagation-and-gradient-descent) |
| **Fig 55** | `fig_55_cbow_gradient_descent_update_rules.png` | Gradient descent parameter update rules | [Backpropagation and Gradient Descent](../README.md#backpropagation-and-gradient-descent) |
| **Fig 56** | `fig_56_word_embeddings_matrix_w1.png` | Extracting word vectors from columns of weight matrix $W_1$ | [Extracting Word Embedding Vectors](../README.md#extracting-word-embedding-vectors) |
| **Fig 57** | `fig_57_word_embeddings_matrix_w2.png` | Extracting word vectors from rows of weight matrix $W_2$ | [Extracting Word Embedding Vectors](../README.md#extracting-word-embedding-vectors) |
| **Fig 58** | `fig_58_word_embeddings_matrix_w3_average.png` | Extracting word vectors by averaging $W_1$ and $W_2^T$ | [Extracting Word Embedding Vectors](../README.md#extracting-word-embedding-vectors) |
| **Fig 59** | `fig_59_intrinsic_evaluation_word_analogies.png` | Intrinsic evaluation using semantic word analogies | [Intrinsic Evaluation](../README.md#intrinsic-evaluation) |
| **Fig 60** | `fig_60_intrinsic_evaluation_clustering_pca.png` | Clustering and PCA projection of embedding representations | [Intrinsic Evaluation](../README.md#intrinsic-evaluation) |
| **Fig 61** | `fig_61_intrinsic_evaluation_cluster_visualization.png` | 2D visualization of semantic word clusters | [Intrinsic Evaluation](../README.md#intrinsic-evaluation) |
