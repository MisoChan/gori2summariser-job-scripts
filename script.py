import torch
import librosa
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM, pipeline


def load_model_gemma(huginng_face_token):
    """Gemma2-9bモデルをローカルでロード"""
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-jpn-it")
    model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-jpn-it",
    torch_dtype=torch.bfloat16,
    token=huginng_face_token # 追加
    ).to("cuda")

    return model, tokenizer

def transcribe_audio_whisper(audio_file):
    """Whisperモデルを使って音声ファイルを文字起こし"""
    model_name = "openai/whisper-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    whisper = pipeline("automatic-speech-recognition",model_name , torch_dtype=torch.float16, device="cuda")

    transcription = whisper(audio_file, return_timestamps=True)
    return transcription["text"],transcription["chunks"]

def important_text(text,timestamp, model, tokenizer):
    """要点を絞り出す"""
    input_text = """
以下の原文を要約してください：
1. 内容を簡潔に要約としてまとめる
2. 重要なポイントを「要点」として抜粋する
3. 長さは元の文の50%以内に収める
4. 見出しは 「##」を必ず使用する。それ以外の出力は許可しない。
5. 「**原文の要約**」という文字列の出力は禁止
5. 原文の出力は禁止
6. URGENT!! 以下のフォーマットを厳守する。それ以外の出力は許可しない。



## 要約
* {要約の内容}

## 要点
* {要点の内容}
* {要点の内容}
~ 
* {要点の内容}


* 原文
    \" """ + text+ " \" "
    
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**input_ids, max_new_tokens=5000) 
    llm_result = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text):]
    ## summary = re.sub(r'##.*\n', '',llm_result )
    summary = summary = re.sub(r'^\s*\n', '', llm_result, flags=re.MULTILINE)
    return {"timestamp": timestamp,"summary": summary}


def combine_chunks(chunks, max_length=2500):
    """チャンクを指定文字数でまとめる関数"""
    combined_chunks = []
    current_chunk = {
        'timestamp': (chunks[0]['timestamp'][0], chunks[0]['timestamp'][1]),
        'text': chunks[0]['text']
    }

    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        
        # 現在のチャンクのテキストがmax_length未満なら結合
        if (next_chunk['timestamp'][0] != 0) and len(current_chunk['text']) + len(next_chunk['text']) <= max_length:
            current_chunk['text'] += " " + next_chunk['text']
            # 終了時間（end）を更新
            current_chunk['timestamp'] = (current_chunk['timestamp'][0], next_chunk['timestamp'][1])
        else:
            # テキストがmax_lengthに達したら保存
            combined_chunks.append(current_chunk)
            # 新しいチャンクの開始（このとき、startとendの両方を更新）
            current_chunk = {
                'timestamp': (next_chunk['timestamp'][0], next_chunk['timestamp'][1]),
                'text': next_chunk['text']
            }
    
    # 最後のチャンクを追加
    combined_chunks.append(current_chunk)
    
    return combined_chunks


def main(audio_file,hug_token):
    """メイン処理"""
    # Whisperモデルのロード
    #whisper_model, whisper_tokenizer = load_model_whisper()

    # 音声ファイルから文字起こし
    text,chunks = transcribe_audio_whisper(audio_file)
    torch.cuda.empty_cache()
    if text:
        print(f"Transcribed Chunks:\n{chunks}")
        print(combine_chunks(chunks))
        

        # 文字起こし結果をテキストファイルに保存
        #save_to_file("transcription.txt", text)
        
        # Gemma2-9bモデルのロード
        gemma_model, gemma_tokenizer = load_model_gemma(hug_token)

        # 要約処理
        split_chunks = combine_chunks(chunks)
        summaries = [important_text(part["text"],part["timestamp"],gemma_model, gemma_tokenizer) for part in split_chunks]
        # 要約結果を表示
        [print(summary['summary']) for summary in summaries]
    else:
        print("文字起こしに失敗しました。")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py audio_file hugging_face_API_key")
        sys.exit(1)

    audio_file = sys.argv[1]
    hug_token = sys.argv[2]
    main(audio_file,hug_token)
