# Name Entity Recognition using Bert Language Model

Use my model:

```python
from transformers import pipeline

model_checkpoint = "Shouhardik/bert-finetuned-ner4"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
token_classifier("My name is Shouhardik, an MSCS student at UCSD!")


