from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Too flaky right now (Feb 2024).  Segfaults, memory exhaustion, etc.
# https://developer.apple.com/metal/pytorch/
# https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+module%3A+mps
#
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print(torch.ones(1, device=device))


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]


def _estimate_sentiment(news):
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]


if __name__ == "__main__":
    tensor, sentiment = _estimate_sentiment(['markets responded negatively to the news!', 'traders were displeased!'])
    print(tensor, sentiment)
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    print(f"torch.backends.mps.is_available()={torch.backends.mps.is_available()}")
