import os
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

import data_processing

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class TextClassification:
    def __init__(self, X, y, label_encoder):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        self.label_encoder = label_encoder
        self.y = self.label_encoder.transform(y)
        self.dataset = self._prepare_dataset(X, self.y)

    def _prepare_dataset(self, X, y):
        encodings = self.tokenizer(X, truncation=True, padding='longest', max_length=512)
        return TextDataset(encodings, y)


def inference(X, y, text_source):
    # Load the label encoder that was saved during training
    label_encoder = torch.load(f'bert_assets/{text_source}/label_encoder.pth')

    # Prepare the dataset for inference using the TextClassification class
    classifier = TextClassification(X, y, label_encoder)
    
    # Load the model
    model = BertForSequenceClassification.from_pretrained(f'bert_assets/{text_source}')
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = Trainer(model=model, args=TrainingArguments(output_dir='bert_assets'))
    
    # Run inference
    predictions = trainer.predict(classifier.dataset)
    predicted_labels = classifier.label_encoder.inverse_transform(predictions.predictions.argmax(-1))
    
    return predicted_labels

def evaluate_model(dataset, text_source):
    filepath = f'data/{dataset}.json'
    print(f'Classifying {text_source}s in {filepath}...')

    # Load and split the data
    X, y = data_processing.preprocess_data(filepath, text_source)

    if dataset == 'all_hand_labeled':
        _, _, X_test, _, _, y_test = data_processing.split_data(X, y)
    elif dataset == 'quora_insincere_hand_labeled':
        X_test = X
        y_test = y

    # Run inference to get the model's predictions on the test set
    predictions = inference(X_test, y_test, text_source)

    # Calculate and print the model's accuracy
    correct_predictions = sum(pred == true for pred, true in zip(predictions, y_test))
    accuracy = correct_predictions / len(y_test)
    print(f'Accuracy: {accuracy*100:.2f}%')

if __name__ == '__main__':
    # Disable tokenizers parallelism to avoid a warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    evaluate_model('all_hand_labeled', 'response')
    print()
    evaluate_model('quora_insincere_hand_labeled', 'prompt')
