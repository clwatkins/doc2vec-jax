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
--window_size 5 \
--vocab_size 50_000
```

## Train model

```bash
python -m doc2vec.train \
--training_data_dir $HOME/doc2vec_data/aclImdb/train/unsup \
--dataset_name imdb_unsup \
--model_dir $HOME/doc2vec_models \
--window_size 5 \
--vocab_size 50_000 \
--batch_size 256 \
--embedding_size 128
```
