import React, { useState } from "react";
import API from "../api";

export default function AudioUploader() {
  const [prediction, setPrediction] = useState(null);
  const [waveformImg, setWaveformImg] = useState(null);
  const [specImg, setSpecImg] = useState(null);

  const handleFile = async (event) => {
    const file = event.target.files[0];
    
    const formData = new FormData();
    formData.append("file", file);

    const res = await API.post("/predict", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    setPrediction(res.data.prediction);
    setWaveformImg(`data:image/png;base64,${res.data.waveform}`);
    setSpecImg(`data:image/png;base64,${res.data.spectrogram}`);
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Clasificador de Enfermedades Cardíacas</h2>

      <input type="file" accept="audio/wav" onChange={handleFile} />

      {prediction && (
        <h3 style={{ marginTop: 20 }}>Predicción: {prediction}</h3>
      )}

      {waveformImg && (
        <>
          <h4>Waveform</h4>
          <img src={waveformImg} alt="waveform" width="400" />
        </>
      )}

      {specImg && (
        <>
          <h4>Espectrograma</h4>
          <img src={specImg} alt="spectrogram" width="400" />
        </>
      )}
    </div>
  );
}
