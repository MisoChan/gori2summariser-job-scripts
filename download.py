from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq

model_name = "openai/whisper-medium"
llm_model_name  = "google/gemma2-9b-it"
# Whisper用のモデルとトークナイザーをロード
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
# model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)