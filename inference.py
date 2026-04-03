from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

REPO_ID = "theinferenceloop/vektor-guard-v1"


def load_model(repo_id: str = REPO_ID):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval()
    model.to(device)

    print(f"Loading model on {device}")

    return tokenizer, model

def predict(text: str, tokenizer, model, device) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    probs = torch.softmax(output.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1)
    confidence = probs[0][predicted_class].item()

    label = "INJECTION" if predicted_class == 1 else "CLEAN"

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "class_id": predicted_class
    }

def main():
    tokenizer, model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nVektor-Guard Inference - type a prompt and press Enter")
    print(f"Type 'quit' to exit\n")

    while True:
        text = input("Prompt> ").strip()

        if text.lower() == "quit":
            break

        if not text:
            continue

        result = predict(text, tokenizer, model, device)

        print(f"    Label:  {result['label']}")
        print(f"    Confidence: {result['confidence']:.2%}")
        print()

if __name__ == "__main__":
    main()