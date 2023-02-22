# Pytorch_Transformer
Transformer from scratch in pytorch

The codes are written following this [tutorial](https://youtu.be/U0s0f995w14)

# The dimensions of query, key and value in multi-headed attention mechanism

In the multi-headed attention mechanism of the Transformer, the query, key, and value vectors are all multi-dimensional tensors, where the dimensionality is determined by the model hyperparameters. Let's denote the dimensionality of the query, key, and value vectors as $d_q$, $d_k$, and $d_v$, respectively.

Suppose we have h attention heads, which means that we have h sets of learnable parameters for the query, key, and value projection matrices, denoted as $W_q^i$`, $W_k^i$, and $W_v^i$ for the i-th head. These projection matrices are used to project the input sequence into h different subspaces. Then, for each head i, we can compute the attention weights and output as follows:

Query, Key, and Value projection:
$q_i = XW_q^i$, where X is the input sequence of shape (batch_size, sequence_length, d_model)
$k_i = XW_k^i$
$v_i = XW_v^i$
Here, d_model is the dimensionality of the input sequence, which is assumed to be the same as $d_q$, $d_k$, and $d_v$.
Scaled Dot-Product Attention:
$score_i = (q_i \cdot k_i^T) / \sqrt{d_k}$, where * denotes matrix multiplication and ^T denotes matrix transpose.
$attention_i = softmax(score_i) \cdot v_i$, where softmax is applied along the sequence length dimension.
Here, score_i is the dot product between the i-th query and key matrices, scaled by the square root of the dimensionality of the key vectors. $attention_i$ is the weighted sum of the value vectors, where the attention weights are determined by the softmax of the dot product between the query and key matrices.
The resulting attention output Z is obtained by concatenating the h attention outputs and projecting the concatenated tensor with a learnable projection matrix $W_o$, where $Z = [attention_1; attention_2; ...; attention_h]W_o$, and ; denotes concatenation along the sequence length dimension.

As an example, let's suppose that we have $d_q = d_k = d_v = 64$, and $h = 8$. Then, the query, key, and value projection matrices have shape (64, 8*64), (64, 8*64), and (64, 8*64), respectively. The attention output for each head has shape (batch_size, sequence_length, 64). After concatenation, the resulting attention output has shape (batch_size, sequence_length, 8*64), which is then projected to the output dimensionality of the model, say, 512, by a learnable projection matrix.
