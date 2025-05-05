from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

app = Flask(__name__)
CORS(app)

# Load pre-trained ASR model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

# Phoneme dictionary (simplified example of expected phonemes)
PHONEME_MAP = {
    "hello": ["HH", "AH", "L", "OW"],
    "world": ["W", "ER", "L", "D"]
}

# Dummy phoneme extraction and error detection (placeholder)
def extract_and_compare_phonemes(transcript):
    words = transcript.lower().split()
    phoneme_errors = []

    for word in words:
        expected_phonemes = PHONEME_MAP.get(word, ["?"])
        # Simulate phoneme error detection
        if "?" in expected_phonemes:
            phoneme_errors.append(f"Unknown phoneme for word '{word}'")

    return phoneme_errors

@app.route("/analyze", methods=["POST"])
def analyze_pronunciation():
    if 'audio' not in request.files:  # Updated key to 'audio' to match frontend
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if not audio_file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            audio_file.save(temp.name)
            speech_array, sampling_rate = torchaudio.load(temp.name)
            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
                speech_array = resampler(speech_array)

            input_values = processor(
                speech_array.squeeze().numpy(),
                return_tensors="pt",
                sampling_rate=16000
            ).input_values

            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = processor.decode(predicted_ids[0])

            phoneme_errors = extract_and_compare_phonemes(transcript)
            os.remove(temp.name)

        return jsonify({
            "transcript": transcript,
            "phoneme_errors": phoneme_errors if phoneme_errors else ["No pronunciation errors detected."]
        })

    except Exception as e:
        if os.path.exists(temp.name):
            os.remove(temp.name)
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)