import click
import numpy as np
import os


def _random_vector(dim):
    return np.random.normal(loc=0, scale=1., size=dim)


@click.command()
@click.option('--input_filename')
@click.option('--output_filename')
@click.option('--seed', type=int, default=3849)
def convert_vectors_into_npy(input_filename: str, output_filename: str,
                             seed: int):
    vectors = []
    tokens = []
    embedding_size = None
    with open(input_filename, 'r') as f:
        for i, l in enumerate(f):
            values = l.split(sep=' ')
            token = values[0]
            vector_s = values[1:]
            if i == 0:
                embedding_size = len(vector_s)

            if len(vector_s) != embedding_size:
                import pdb
                pdb.set_trace()
            assert len(vector_s) == embedding_size, 'invalid vector size'

            vector = np.array([float(x) for x in vector_s], dtype=np.float32)

            tokens.append(token)
            vectors.append(vector)

    np.random.seed(seed)
    start_vector = np.abs(_random_vector(embedding_size))
    end_vector = -np.abs(_random_vector(embedding_size))

    vectors.append(start_vector)
    vectors.append(end_vector)

    tokens_array = np.array(tokens)
    embeddings = np.stack(vectors)

    with open(output_filename, 'wb') as f:
        np.savez(f, tokens=tokens_array, embeddings=embeddings)


if __name__ == '__main__':
    convert_vectors_into_npy()
