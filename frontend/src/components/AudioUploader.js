import React, { useState } from "react";
import API from "../api";
import "./AudioUploader.css";

const diseaseInfo = {
  'AS': 'Estenosis Aórtica: Estrechamiento de la válvula aórtica que dificulta el flujo sanguíneo.',
  'MR': 'Regurgitación Mitral: La válvula mitral no cierra correctamente, causando reflujo de sangre.',
  'MS': 'Estenosis Mitral: Estrechamiento de la válvula mitral que restringe el flujo sanguíneo.',
  'MVP': 'Prolapso de Válvula Mitral: Las valvas de la válvula mitral sobresalen hacia la aurícula.',
  'N': 'Normal: No se detectaron anomalías cardíacas significativas.'
};

export default function AudioUploader() {
  const [prediction, setPrediction] = useState(null);
  const [waveformImg, setWaveformImg] = useState(null);
  const [specImg, setSpecImg] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState(null);

  const handleFile = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setFileName(file.name);
    setLoading(true);
    setPrediction(null);
    setWaveformImg(null);
    setSpecImg(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await API.post("/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPrediction(res.data.prediction);
      setWaveformImg(`data:image/png;base64,${res.data.waveform}`);
      setSpecImg(`data:image/png;base64,${res.data.spectrogram}`);
    } catch (error) {
      console.error("Error al procesar el archivo:", error);
      alert("Error al procesar el archivo de audio. Por favor, intenta nuevamente.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="uploader-container">
      <div className="uploader-card">
        <div className="header-section">
          <h1 className="main-title">Clasificador de Enfermedades Cardíacas</h1>
          <p className="subtitle">
            Análisis mediante Inteligencia Artificial de sonidos cardíacos para detección temprana
          </p>
        </div>

        <div className="upload-section">
          <label htmlFor="audio-file" className="upload-label">
            <div className="upload-icon">🎵</div>
            <p className="upload-text">
              {loading ? "Procesando..." : "Seleccionar Archivo de Audio"}
            </p>
            <p className="upload-hint">Formato soportado: WAV</p>
            {fileName && !loading && (
              <div className="file-selected">
                Archivo seleccionado: {fileName}
              </div>
            )}
            {loading && (
              <>
                <div className="loading-spinner"></div>
                <p className="loading-text">Analizando audio cardíaco...</p>
              </>
            )}
          </label>
          <input
            id="audio-file"
            type="file"
            accept="audio/wav"
            onChange={handleFile}
            className="file-input"
            disabled={loading}
          />
        </div>

        {prediction && (
          <div className="prediction-section">
            <p className="prediction-label">Diagnóstico detectado</p>
            <h2 className="prediction-result">{prediction}</h2>
            <div className="disease-info">
              {diseaseInfo[prediction] || "Información no disponible"}
            </div>
          </div>
        )}

        {(waveformImg || specImg) && (
          <div className="visualizations-section">
            {waveformImg && (
              <div className="visualization-card">
                <h3 className="visualization-title">Forma de Onda</h3>
                <img
                  src={waveformImg}
                  alt="Forma de onda del audio cardíaco"
                  className="visualization-image"
                />
              </div>
            )}

            {specImg && (
              <div className="visualization-card">
                <h3 className="visualization-title">Espectrograma</h3>
                <img
                  src={specImg}
                  alt="Espectrograma del audio cardíaco"
                  className="visualization-image"
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
