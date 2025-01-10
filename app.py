from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")  # Folder where templates are stored

# Load the model and tokenizer (same as before)
MODEL_PATH = "./model.safetensors"
TOKENIZER_PATH = "./"
CONFIG_PATH = "./config.json"
GENERATION_CONFIG_PATH = "./generation_config.json"
SPECIAL_TOKENS_MAP_PATH = "./special_tokens_map.json"
TOKENIZER_CONFIG_PATH = "./tokenizer_config.json"

try:
    # Load tokenizer with local configurations
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        config=CONFIG_PATH,
        tokenizer_config_file=TOKENIZER_CONFIG_PATH,
        special_tokens_map_file=SPECIAL_TOKENS_MAP_PATH,
    )

    # Load model with local configurations
    model = AutoModelForSeq2SeqLM.from_pretrained(
        TOKENIZER_PATH,
        config=CONFIG_PATH,
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
except Exception as e:
    raise RuntimeError(f"Error loading the model or tokenizer: {e}")

# Define input schema for summarization
class SummarizationInput(BaseModel):
    text: str

# Route for the main page (similar to Flask's `index`)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route for handling prediction requests (similar to Flask's `/predict`)
@app.post("/predict")
async def predict(gender: str = Form(...), salary: int = Form(...)):
    # Handle gender as in the Flask example
    if gender == 'male':
        gender = 1
    else:
        gender = 0

    # Here, using a mock model for illustration (replace this with your actual model)
    model = pickle.load(open('model14.pkl', 'rb'))
    decision = model.predict(np.array([gender, salary]).reshape(1, 2))

    if decision[0] == 1:
        decision = 'purchased'
    else:
        decision = 'not purchased'
    
    return {"decision": decision}

# Route for summarization (similar to your FastAPI summarization route)
@app.post("/summarize/")
async def summarize(input_data: SummarizationInput):
    try:
        # Tokenize the input text
        inputs = tokenizer(
            input_data.text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Generate the summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Run the app (FastAPI uses Uvicorn as the ASGI server)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
