# Usage

Basically, you run `main.py` to train your model. There are predefined templates: `train_bert`, `train_dae`, `train_vae_search_beta`, and `train_vae_give_beta`. `main.py` will ask you which dataset to train on (ML-1m or ML-20m) and whether to run test set inference (y/n) for the current run at the end of training.

Here are a few examples that will help you grasp the usage.

### Train BERT4Rec on ML-20m and run test set inference after training

```bash
printf '20\ny\n' | python main.py --template train_bert
```

### Search for optimal beta for VAE on ML-1m and do not run test set inference

```bash
printf '1\nn\n' | python main.py --template train_vae_search_beta
```

Note that for `train_vae_give_beta`, you must specify the optimal beta value in `templates.py`.
