import os
import uuid
import numpy as np
import librosa

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from joblib import load

app = FastAPI()

# ==============================
# Health Check
# ==============================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Parkinson API"
    }

# ==============================
# Load Model
# ==============================

model = load("./parkinsons_model.joblib")
scaler = load("./scaler.joblib")

# ==============================
# Predict Endpoint Schema
# ==============================

class ParkinsonInput(BaseModel):
    features: list

# ==============================
# Extract Features
# ==============================

def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=16000)

    features = []

    # Pitch
    pitches, magnitudes = librosa.piptrack(
        y=y,
        sr=sr
    )

    pitch_values = pitches[
        magnitudes > np.median(magnitudes)
    ]

    features += [

        float(np.mean(pitch_values))
        if len(pitch_values) > 0 else 0.0,

        float(np.max(pitch_values))
        if len(pitch_values) > 0 else 0.0,

        float(np.min(pitch_values))
        if len(pitch_values) > 0 else 0.0,
    ]

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)

    features += [
        float(np.mean(zcr)),
        float(np.std(zcr)),
        float(np.min(zcr))
    ]

    # Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features += [
        float(np.mean(spec_cent)),
        float(np.std(spec_cent)),
        float(np.min(spec_cent))
    ]

    # Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features += [
        float(np.mean(spec_bw)),
        float(np.std(spec_bw)),
        float(np.min(spec_bw))
    ]

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    features += [
        float(np.mean(rolloff)),
        float(np.std(rolloff)),
        float(np.min(rolloff))
    ]

    # MFCC
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=3
    )

    features += [
        float(np.mean(mfccs[i]))
        for i in range(3)
    ]

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    features += [
        float(np.mean(chroma)),
        float(np.std(chroma)),
        float(np.min(chroma)),
        float(np.max(chroma))
    ]

    features = features[:22]

    if len(features) != 22:
        raise Exception(
            f"Expected 22 features, got {len(features)}"
        )

    return features

# ==============================
# Predict Using Features
# ==============================

@app.post("/predict")
def predict(data: ParkinsonInput):

    x = np.array(
        data.features
    ).reshape(1, -1)

    x_scaled = scaler.transform(x)

    prediction = int(
        model.predict(x_scaled)[0]
    )

    probability = float(
        model.predict_proba(x_scaled)[0][1]
    )

    return {
        "prediction": prediction,
        "probability": probability
    }

# ==============================
# Analyze Audio File
# ==============================

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):

    allowed_extensions = (
        ".wav",
        ".mp3",
        ".m4a"
    )

    if not file.filename.lower().endswith(
        allowed_extensions
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only WAV, MP3 and M4A files are allowed"
            )
        )

    extension = os.path.splitext(
        file.filename
    )[1]

    temp_filename = (
        f"temp_{uuid.uuid4().hex}"
        f"{extension}"
    )

    try:

        # Save uploaded file
        with open(
            temp_filename,
            "wb"
        ) as buffer:

            buffer.write(
                await file.read()
            )

        # Extract Features
        features = extract_features(
            temp_filename
        )

        # Scale Features
        x = np.array(
            features
        ).reshape(1, -1)

        x_scaled = scaler.transform(
            x
        )

        # Predict
        prediction = int(
            model.predict(
                x_scaled
            )[0]
        )

        probability = float(
            model.predict_proba(
                x_scaled
            )[0][1]
        )

        return {
            "prediction": prediction,
            "probability": probability
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if os.path.exists(
            temp_filename
        ):
            os.remove(
                temp_filename
            )