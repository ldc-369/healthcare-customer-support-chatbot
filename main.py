from flask import Flask, request, jsonify, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import torch

app = Flask(__name__)

# load model
model = T5ForConditionalGeneration.from_pretrained("./healthcare-customer-support-chatbot/model")
tokenizer = T5Tokenizer.from_pretrained("./healthcare-customer-support-chatbot/model")

device = model.device
model.eval()  # chuyển sang inference mode


def clean_text(text):
    text = re.sub(r'\r\n', ' ', text)       # "\n" thành " "
    text = re.sub(r'\s+', ' ', text)        # Khoảng trắng thành ""
    text = re.sub(r'<.*?>', '', text)       # Loại bỏ thẻ
    text = text.strip().lower()
    return text


def generate_response(query):
    query = clean_text(query)

    inputs = tokenizer(
        query,
        padding="max_length",
        truncation=True,
        max_length=250,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=250,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


@app.route('/', methods=["GET"])
def render_index():
    return render_template('index.html')


@app.route("/message", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"status": False, "message": "Please enter a message!!!"}), 400
    res = generate_response(user_message)
    
    return jsonify({"status": True, "res": res}), 200


if __name__ == "__main__":
    app.run(debug=True)