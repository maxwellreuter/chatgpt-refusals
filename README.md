# I'm Afraid I Can't Do That - Predicting Prompt Refusal in Black-Box Generative Language Models
This repository contains datasets and code used to obtain the results presented in [the paper](https://arxiv.org/pdf/2306.03423.pdf).

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
Omit the `--fit_random_forest_on_quora_10k` flag to skip fitting a random forest prompt classifier on the Bootstrapped Quora Insincere Questions dataset. This fit may take a long time to complete due to the large number of estimators. The rest of the script takes about 20 seconds.
```sh
> python classical_model_results.py --fit_random_forest_on_quora_10k

[Table 4]
Calculating classical model accuracies for dataset: Hand-Labeled...
Achieved 90.62% test accuracy in response classification with LogisticRegression.
Achieved 86.72% test accuracy in response classification with RandomForestClassifier.
Achieved 73.91% test accuracy in prompt classification with LogisticRegression.
Achieved 72.18% test accuracy in prompt classification with RandomForestClassifier.

[Fig. 3]
Calculating n-gram coefficients for dataset: Hand-Labeled...

[Fig. 4]
Calculating n-gram coefficients for dataset: Bootstrapped Quora Insincere Questions...

Finished; n-gram coefficients were written to the "results" folder.
```

## Reproduce BERT classification results
Download `bert_assets.zip` from [this Google Drive](https://drive.google.com/drive/folders/1ak4IeIYy3XMRSWsRv3WzcVkGhlJlhhSC?usp=sharing) and unzip it in this directory (867MB uncompressed).

This process should take about 1 minute on CPU:
```sh
> python bert_results.py

Classifying responses in data/all_hand_labeled.json...
100%|███████████████████████████████████████████| 33/33 [00:48<00:00,  1.46s/it]
Accuracy: 92.31%

Classifying prompts in data/quora_insincere_hand_labeled.json...
100%|█████████████████████████████████████████| 127/127 [00:31<00:00,  4.01it/s]
Accuracy: 75.52%
```

## Programmatically querying ChatGPT
We queried ChatGPT via the [OpenAI Chat API](https://platform.openai.com/docs/guides/chat). To do this, we used the model code `gpt-3.5-turbo-0301` (see [here](https://platform.openai.com/docs/models/gpt-3-5) for details). The total cost for all of our queries was roughly $50.
