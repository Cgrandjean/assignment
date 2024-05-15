from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_from_disk
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding


label2id = {'noHate':0, 'hate':1}
id2label = {v: k for k, v in label2id.items()}
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased',label2id=label2id,id2label=id2label, num_labels=2)

train_ds=load_from_disk('data/train_ds')
test_ds=load_from_disk('data/test_ds')


def preprocess_function(ex):
    return tokenizer(ex['text'],padding=True, truncation=True, max_length=512)

train_ds = train_ds.map(preprocess_function, batched=True)
test_ds = test_ds.map(preprocess_function, batched=True)



def compute_metrics(preds):
    logits, labels = preds.predictions,preds.label_ids
    probas =softmax(logits, axis=1)[:, 1] 
    preds = logits.argmax(-1)
    precision, recall, f1, supp = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {"precision": list(precision),
            "recall": list(recall),
            "accuracy": acc,
            "f1": list(f1),
            "roc_auc":roc_auc_score(labels,probas)
            }


training_args = TrainingArguments(
    output_dir='./models/Bert-trained/',  
    num_train_epochs=5,              
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=16,   
    warmup_steps=300,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=50,
    eval_steps=50,
    evaluation_strategy="steps",
    learning_rate=2e-5,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1,              
    load_best_model_at_end=True,    
    metric_for_best_model='eval_loss',   
    greater_is_better=False         

    # device='cuda' 
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics 

)

trainer.train()

results = trainer.evaluate()
print(results)
model.save_pretrained('./models/Bert-trained/Best')