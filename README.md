Attention Is All You Need Paper Implementation
==============================================

This is an unaffiliated implementation of the following paper:

    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I., 2017. Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).


### Motivation

Prior to the wave of transformer-style attention in the NLP and wider deep learning communities, language models 
were moving much more towards applying recurrent networks to tackle complex linguistic tasks.
In particular, for the class of problems that take an input sequence and transform it into an output
sequence (known as sequence transduction), recurrent networks were the favored approach. 
The main motivation behind the paper was the observation that although recurrent networks 
as they were formulated could capture long-term relationships, there was usually a hard limit 
on how far back information would propagate due to the vanishing gradient problem. Another limiting factor
when working with long sequences is that recurrent networks pass information sequentially which can be
inefficient.

The paper introduced the Transformer model, which uses a special kind of global attention that
is able to transform an input sequence of length N into an output sequence of length M, using O(M)
forward passes, where each forward pass can attend to the inputs and previously produced outputs in parallel.
By structuring learning in this way, some of the aforementioned problems that faced
recurrent models could be greatly diminished.

### Key Contributions

* First neural sequence transduction model that worked entirely on self-attention.
* Outperformed all previous models on the WMT 2014 English-to-German and English-to-French tasks.

### Approach

As described, the main goal of the Transformer is to take an input sequence of tokens (i.e. words or subword units)
and transform them into an output sequence of tokens. A sequence of tokens, however, can be represented
in any number of different ways and so the first step in this process requires that a
suitable representation be chosen. 

#### Input/Output Representation

For the Transformer model, it is expected
that the input token sequence (i.e. a sentence) is converted into a sequence of vectors known as embeddings.
Such embeddings have been shown to be powerful ways to capture semantic meaning
in a vector space. This is typically done by running a separate training process, using
methods such as Word2Vec or GloVe, to learn a representation in which tokens that
co-occur are closer in the vector space. There are many pre-trained token (i.e. word) embeddings
that can be used off-the-shelf, though they can also be generated using existing tools relatively
easily as well. In sum, this pre-processing step takes an input sequence of tokens
(which are often words but can also be subword units) and uses what is effectively a learned
lookup table to convert that into a sequence of vectors of a common dimensionality (i.e. 256).
This can be represented as a 2D matrix of shape [number of input tokens, input embedding dimensionality].
In the paper this sequence of vectors is described as the `Input Embeddings`.

In much the same way, the outputs predicted by the Transformer are in the form of discrete categorical distributions over the possible output tokens. Each of these output tokens have a corresponding embedding 
vector which could have be trained using entirely different data as the input embeddings 
(usually another language).

Additionally, there is one important consideration that should be made when using token embeddings.
With sequence transducers, there are often a set of special tokens that are interpreted
as signals to either begin or end a sequence. Specifically in the case of the Transformer, before
producing any outputs, the output embeddings are initialized using a special "begin" embedding. 
Subsequent predicted embeddings are then concatenated with this "begin" embedding to produce the 
output embeddings for future steps. The Transformer also signals that it is finished with
sequence transduction by predicting the special "end" token as the most likely next token.

Although the Transformer does not maintain a hidden state that gets updated sequentially
like an RNN, it still generates the sequence of output tokens autoregressively one at a time.
What this means is that to generate M output tokens requires running M different forward
passes. This is done by first shifting the previously generated output tokens to the right typically
by inserting a start/begin token at the beginning as described earlier. These tokens are then
converted into embeddings (described in the paper as `Output Embedding`) which are continuous
vector representations. The model then uses the input embeddings along with these shifted output 
embeddings in order to generate a next token probability distribution for each of the shifted output 
positions. So after having produced `i` output tokens, the model will construct the new shifted output 
embeddings of length `i+1`. For each of these `i+1` embeddings, there will be a
discrete categorical distribution corresponding to the next token probabilities. The `i+1`th distribution
can then be used to predict the next token in the output sequence (i.e. taking the most likely or
by performing beam search). This structure of inputs and outputs makes end-to-end neural
learning possible and enables the network architectures described in the following sections.

#### Encoder/Decoder Architecture

The Transformer's neural architecture can be decomposed into two components, the encoder and
the decoder. The encoder is a neural module that takes as input the previously described input 
embeddings with a separate positional embedding added to it. A positional embedding is just
another vector of the same dimensionality as the token embedding it is used with, but that
contains information specific to the position in the input sequence it is associated with.
The paper describes the exact positional embedding used, but note that it is not a 
learned embedding and is just computed using a function of the position in the sequence and 
the size of the embedding space. The encoder produces a sequence of encodings with the same
length as the input. For every input sequence, the encoder is run only a single time.
The decoder, on the other hand is, a separate neural module that unlike the encoder is run M 
times, once for each output token. As described earlier, the output of the decoder is a sequence of
discrete categorical distributions over the set of possible next output tokens for each output position. 
Furthermore, there are two inputs to the decoder, the encodings (from the encoder) and the 
aforementioned output embeddings with positional embeddings added to them just like with the encoder 
inputs.

#### Multi-Head Attention

Both the encoder and the decoder heavily use what is known as multi-head scaled dot-product attention,
or multi-head attention (MHA).
This attention mechanism is used in cases where given three sequences of vectors, Q, K, and V of
lengths L, J, and J respectively, you want to generate an output sequence of vectors Z of length L in such a way
that each output vector is produced by attending over all of the J elements in sequence K using each
of the L elements in sequence Q. Additionally, the output vector should be produced mainly by "looking" at 
the vectors in the sequence V. The matrices Q, K, and V are dubbed queries, keys, and values respectively
because this mechanism can be described as using using each query to lookup values using the keys to finally
computing an output. Dot product attention accomplishes this by first linearly mapping the 
queries (Q) of length L and keys (K) of length J into vectors with the same dimensionality.
Additionally, the J values in V are linearly mapped to vectors with the desired dimensionality of the output.
Each of the J output vectors of the sequence Z are then computed using a convex 
combination of these mapped values. The weights of the convex combination for the output at position
`i` are generated by first taking the dot-product of the linearly mapped query at position `i` with all of the
linearly mapped keys. This produces J weights, one corresponding to each of the keys. These weights are then
scaled using a constant that depends on the dimensionality used for the queries and keys in order to avoid 
oversaturation of a softmax applied right afterwards to produce the final convex combination weights. 
Then as mentioned, the output
is computed by taking a combination of the J linearly mapped values using these weights.
This process is done for each of the L queries producing L outputs. All of this can be accomplished
efficiently using solely matrix operations, which are detailed in the paper. Furthermore,
by applying this multiple times using different parameters and stacking the outputs, you arrive
at the final multi-headed attention (MHA) mechanism.

MHA is used in the encoder for self-attention. Specifically, all three input sequences, Q, K, and V,
used earlier are the same in this case and correspond to the encoder inputs. The decoder
applies MHA in two ways, both for self-attention on the summed output and positional embeddings
as well as for attention using the encodings as the queries and keys while the decoder's
self-attention as the values. One important detail was that the self-attention used by the
decoder applied masking of the dot-product weights so that the convex combination for the output
at position `i` would never include information at later time steps. This enforces the autoregressive
nature of the model.

#### Further Details

Along with MHA, feed forward networks were used as well as layer normalization and residual connections.
These operations were combined into stacks which were repeated multiple times to produce deep
models. The decoder also had a final linear layer with a softmax applied to generate the
predicted next token probabilities for each of the shifted outputs.

See the paper for more details.

### Implementation

For this implementation, tokenization was performed using byte-pair encoding (BPE) and GloVe was used to
generate the word embeddings. Different token embeddings were generated for the source and target languages. Additionally, the model size and embedding dimensionality used was 256 rather than 512 as in the paper. The Adam optimizer with the same hyperparameters
and learning rate schedule as described in the paper was also used.

There are scripts to download the dataset and generate the BPE files under `/scripts`. 
As the WMT13 dataset doesn't provide a train/val split, one was generated using
`chainer_transformer.tools.generate_trainval_split` before tokenization.
After generating the tokenizations (BPE files) on the training set of the source and target languages, the
[GloVe](https://github.com/stanfordnlp/GloVe) repo was used to generate the word 
embeddings. In that repository, the `demo.sh` script was modified so that the `CORPUS` variable pointed to the
training corpus being processed. Additionally, a few more variable updates were made, with 
`VOCAB_MIN_COUNT=0` and `VECTOR_SIZE=256`. This generated a `vectors.txt` file for
each vocabulary which was transformed into an npz file using the
`chainer_transformer.tools.convert_vectors_into_npy` tool. These fixed vectors were then
passed to `chainer_transformer.trainer` during training.