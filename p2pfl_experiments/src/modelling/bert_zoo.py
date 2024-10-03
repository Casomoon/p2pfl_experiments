# model imports 
from transformers import (BertForSequenceClassification,
                          DistilBertForSequenceClassification,
                          MobileBertForSequenceClassification)
# model dictionary 
bert_model_zoo = {
    "bert": BertForSequenceClassification, 
    "distilbert": DistilBertForSequenceClassification,
    "mobilebert": MobileBertForSequenceClassification
}
# tokenizer imports 
from transformers import (BertTokenizerFast,
                          DistilBertTokenizerFast, 
                          MobileBertTokenizerFast)
# tokenizer dictionary
tokenizer_zoo = {
    "bert": BertTokenizerFast,
    "distilbert": DistilBertTokenizerFast,
    "mobilebert": MobileBertTokenizerFast
}
# pretrained model and tokenizer path for each model 
pretrained_paths  = {
    "bert" : "bert-base-uncased", 
    "distilbert" : "distilbert-base-uncased", 
    "mobilebert" : "google/mobilebert-uncased"
}


def get_bert_by_string(model_type: str, num_labels: int = 2):
    """
    Returns an instance of the specified BERT model with pretrained weights.

    Args:
        model_type (str): Type of the model ('bert', 'distilbert', 'mobilebert').
        num_labels (int): Number of labels for classification.

    Returns:
        Pretrained model instance.
    """
    assert model_type in bert_model_zoo.keys() and model_type in pretrained_paths.keys(), f"Model type '{model_type}' not recognized."
    base_model_class = bert_model_zoo[model_type]
    pretrained_path = pretrained_paths[model_type]
    pretrained_model = base_model_class.from_pretrained(pretrained_path, num_labels = num_labels)
    return pretrained_model

def get_tokenizer_by_string(model_type: str):
    """
    Returns an instance of the tokenizer for the specified model type.

    Args:
        model_type (str): Type of the model ('bert', 'distilbert', 'mobilebert').

    Returns:
        Pretrained tokenizer instance.
    """
    assert model_type in tokenizer_zoo.keys() and model_type in pretrained_paths.keys(), \
        f"Tokenizer for model type '{model_type}' not recognized."
    
    tokenizer_class = tokenizer_zoo[model_type]
    pretrained_path = pretrained_paths[model_type]  # Get the path for the tokenizer
    
    tokenizer = tokenizer_class.from_pretrained(pretrained_path)
    
    return tokenizer