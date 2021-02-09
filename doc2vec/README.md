# Doc2Vec training example

## Generate training data

```bash
python -m doc2vec.generate_training_data \
--training_data_dir $HOME/Documents/doc2vec_data/aclImdb/train/unsup \
--dataset_name imdb_unsup \
--window_size 5 \
--vocab_size 50_000
```

## Train model

```bash
python -m doc2vec.train \
--training_data_dir $HOME/Documents/doc2vec_data/aclImdb/train/unsup \
--dataset_name imdb_unsup \
--model_dir $HOME/Documents/doc2vec_models \
--window_size 5 \
--vocab_size 50_000 \
--batch_size 256 \
--embedding_size 128
```
