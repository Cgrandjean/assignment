from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline,AutoTokenizer,AutoModelForSequenceClassification

app = FastAPI()

# Initialize the pipeline
label2id = {'Nohate':0, 'Hate':1}
id2label = {v: k for k, v in label2id.items()}
tokenizer = AutoTokenizer.from_pretrained('chgrdj/nlp-assignment-Bert')
model = AutoModelForSequenceClassification.from_pretrained('chgrdj/nlp-assignment-Bert', 
                                            label2id=label2id, 
                                            id2label=id2label,
                                            num_labels=len(label2id))
model_pipeline = pipeline('text-classification', model=model,tokenizer=tokenizer)

class Item(BaseModel):
    text: str

@app.post("/predict")
async def predict(item: Item):
    
    result = model_pipeline(item.text)
    return {
        "text": item.text,
        "predictions": result
    }
