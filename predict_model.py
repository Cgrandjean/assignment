
from transformers import pipeline,AutoTokenizer,AutoModelForSequenceClassification
from datasets import load_dataset,load_from_disk

test_ds=load_from_disk('data/test_ds')
test_texts=test_ds['text']

label2id = {'Nohate':0, 'Hate':1}
id2label = {v: k for k, v in label2id.items()}
tokenizer = AutoTokenizer.from_pretrained('chgrdj/nlp-assignment-Bert')
model = AutoModelForSequenceClassification.from_pretrained('chgrdj/nlp-assignment-Bert', 
                                            label2id=label2id, 
                                            id2label=id2label,
                                            num_labels=len(label2id))
model_pipeline = pipeline('text-classification', model=model,tokenizer=tokenizer)

print(model_pipeline(test_texts)[:10])