# I'm Afraid I Can't Do That - Predicting Prompt Refusal in Black-Box Generative Language Models
This repository contains datasets and code used to obtain the results presented in the paper (forthcoming).

To run the code, you will need to install the appropriate dependencies using [`pip`](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-pypi), [Conda](https://docs.conda.io/en/latest/), or similar.

## Datasets
| Dataset | Number of Samples (n) | Filename |
|--------------------------|-----------------------|----------|
| New York Post            | 21                    | `nyp.json` |
| Political Figures        | 700                   | `political_figures.json` |
| Quora Insincere Questions| 1,009                 | `quora_insincere_hand_labeled.json` |
| Hand-Labeled             | 1,730                 | `all_hand_labeled.json` |
| Unlabeled Quora Insincere Questions | 10,000 | `quora_insincere_large_unlabeled.json` |
| Bootstrapped Quora Insincere Questions | 10,000 | `quora_insincere_large_bootstrap.json` |

## Reproduce word importance (n-gram coefficients) results
Omit the `--fit_random_forest_on_quora_10k` flag to skip fitting a random forest prompt classifier on the Bootstrapped Quora Insincere Questions dataset. This fit may take about an hour (or more) to complete due to the large number of estimators. The rest of the script takes about 20 seconds.
```sh
> python classical_model_results.py --fit_random_forest_on_quora_10k

[Table 4 (classical models)]
Calculating classical model accuracies for dataset: Quora Insincere Questions...
Achieved 82.24% test accuracy in response classification using LogisticRegression.
Achieved 76.32% test accuracy in response classification using RandomForestClassifier.
Achieved 73.54% test accuracy in prompt classification using LogisticRegression.
Achieved 71.66% test accuracy in prompt classification using RandomForestClassifier.

[Fig. 3]
Calculating n-gram coefficients for dataset: Hand-Labeled...

[Fig. 4]
Calculating n-gram coefficients for dataset: Bootstrapped Quora Insincere Questions...

Finished - n-gram coefficients were written to the "results" folder.
```

## Reproduce BERT classification results
Download `bert_assets.zip` from [this Google Drive](https://drive.google.com/drive/folders/1ak4IeIYy3XMRSWsRv3WzcVkGhlJlhhSC?usp=sharing) and unzip it in this directory (867MB uncompressed).

This process should take about 1 minute on CPU:
```sh
> python bert_results.py

Classifying responses in data/all_hand_labeled.json...
100%|██████████████████████████████████████████████████████████████| 33/33 [00:50<00:00,  1.53s/it]
Accuracy: 92.31%

Classifying prompts in data/quora_insincere_hand_labeled.json...
100%|██████████████████████████████████████████████████████████████| 19/19 [00:04<00:00,  4.23it/s]
Accuracy: 75.52%
```

## Programmatically querying ChatGPT
We queried the March 1, 2023 snapshot of ChatGPT via the [OpenAI Chat API](https://platform.openai.com/docs/guides/chat). To do this, we used the model code `gpt-3.5-turbo-0301` (see [here](https://platform.openai.com/docs/models/gpt-3-5) for details). The total cost for all of our queries was roughly $50.
