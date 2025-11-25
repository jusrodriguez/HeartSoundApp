import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import time

# ----------------------------
# CONFIGURACIÓN GLOBAL
# ----------------------------
fs = 44100            # Frecuencia de muestreo
CHUNK = 2048          # Bloque de lectura (grande para latidos)
GAIN = 12             # Amplificación
LOW, HIGH = 20, 150   # Banda cardíaca
MIN_HR = 40           # BPM mínimo
MAX_HR = 180          # BPM máximo
S1S2_MIN_DIST = 0.15  # Distancia mínima entre picos S1/S2 (segundos aprox)
# ----------------------------

# --- FILTRO PASA-BANDA ---
def bandpass(low, high, fs, order=4):
    ny = fs / 2
    b, a = butter(order, [low/ny, high/ny], btype='band')
    return b, a

b_bp, a_bp = bandpass(LOW, HIGH, fs)

# --- FILTRO SUAVIZADOR ---
def smooth(signal, w=10):
    return np.convolve(signal, np.ones(w)/w, mode="same")

# --- CONFIGURAR GRAFICA ---
plt.ion()
fig, ax = plt.subplots(figsize=(11,5))
line, = ax.plot(np.zeros(CHUNK))
peak_marks, = ax.plot([], [], 'ro')
ax.set_ylim(-1, 1)
ax.set_xlim(0, CHUNK)
ax.set_title("Monitor Cardíaco Digital (micrófono + estetoscopio)")
ax.set_xlabel("Muestras")
ax.set_ylabel("Amplitud")

# Texto BPM
bpm_text = ax.text(0.75, 0.9, "BPM: --", transform=ax.transAxes, fontsize=14)

print("Grabando y mostrando monitor cardíaco en tiempo real...")
print("Cierra la ventana para detener.")

stream = sd.InputStream(samplerate=fs, channels=1, blocksize=CHUNK, dtype='float32')
stream.start()

# Para cálculo de BPM
last_peaks_times = []

try:
    while True:
        block, _ = stream.read(CHUNK)
        block = block.flatten()

        # --- FILTRO CARDIACO ---
        filtered = filtfilt(b_bp, a_bp, block)

        # --- SUAVIZADO ---
        filtered = smooth(filtered, w=15)

        # --- NORMALIZAR + AMPLIFICAR ---
        max_val = np.max(np.abs(filtered)) or 1
        filtered = (filtered / max_val) * GAIN
        filtered = np.clip(filtered, -1, 1)

        # --- DETECCIÓN DE PICOS (S1/S2) ---
        distance = int(S1S2_MIN_DIST * fs)
        peaks, _ = find_peaks(np.abs(filtered), height=0.2, distance=distance)

        # Guardar timestamps de picos para calcular BPM
        now = time.time()
        for p in peaks:
            last_peaks_times.append(now)

        # Eliminar picos antiguos (más de 5 segundos)
        last_peaks_times = [t for t in last_peaks_times if now - t < 5]

        # Calcular BPM
        if len(last_peaks_times) > 1:
            intervals = np.diff(last_peaks_times)
            avg_interval = np.mean(intervals)
            bpm = int(60 / avg_interval)
        else:
            bpm = "--"

        # --- ACTUALIZAR GRAFICA ---
        line.set_ydata(filtered)

        peak_marks.set_xdata(peaks)
        peak_marks.set_ydata(filtered[peaks])

        bpm_text.set_text(f"BPM: {bpm}")

        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    pass

stream.stop()
print("Monitor detenido.")
