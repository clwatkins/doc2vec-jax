# Doc2Vec training example

## Download training data

```bash
mkdir $HOME/doc2vec_data/
curl https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz --output $HOME/doc2vec_data/aclImdb.tar.gz
tar -xf $HOME/doc2vec_data/aclImdb.tar.gz -C $HOME/doc2vec_data
```

## Generate training data

```bash
python -m doc2vec.generate_training_data \
--training_data_dir $HOME/doc2vec_data/aclImdb/train/unsup \
--dataset_name imdb_unsup \
--architecture dbow \
--window_size 5 \
--vocab_size 50_000 \
--subsampling_thresh 10e-5 \
--num_ns 4
```

## Train model

```bash
mkdir $HOME/doc2vec_models/
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m doc2vec.train \
--training_data_dir $HOME/doc2vec_data/aclImdb/train/unsup \
--dataset_name imdb_unsup \
--model_dir $HOME/doc2vec_models \
--architecture dbow \
--context_mode average \
--window_size 5 \
--vocab_size 50_000 \
--batch_size 256 \
--embedding_size 128 \
--subsampling_thresh 10e-5 \
--num_ns 4
```

Note we disable XLA memory pre-allocation, as that has a tendency to cause issues -- suspect because of its interaction 
with the Tensorflow data loader also reserving GPU memory. See: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html 