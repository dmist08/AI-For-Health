# AI for Health — SRIP 2026 Project Report
**Objective:** Detecting breathing irregularities (Apnea and Hypopnea) from overnight physiological sleep data.

## 1. Introduction and Data Exploration
The provided dataset contained overnight sleep data from 5 participants (AP01–AP05), including:
*   **Nasal Airflow** (32 Hz)
*   **Thoracic Movement** (32 Hz)
*   **SpO2 (Oxygen Saturation)** (4 Hz)
*   **Events Log** and **Sleep Profile**

### Data Alignment and Visualization
Since the sensors operated at different sampling frequencies (32 Hz vs. 4 Hz), the initial challenge was aligning the signals temporally. We accomplished this by standardizing the timestamps across all sequences. `vis.py` generates a comprehensive timeline array for all patients showing the full 8-hour plots. Missing modality edge cases (such as AP01's differing format) were handled securely. No interpolation was used simply for visualization to maintain absolute ground truth.

## 2. Signal Preprocessing & Dataset Creation
To isolate human breathing (10–24 Breaths Per Minute, approx 0.17 Hz–0.40 Hz), we applied digital filtering:
*   **Respiration (Flow & Thoracic):** Applied a 4th-order Butterworth Bandpass filter (0.17 Hz to 0.40 Hz) using SciPy to remove high-frequency noise and extreme low-frequency baseline wander.
*   **SpO2:** Applied a low-pass filter (cutoff 1.0 Hz) because SpO2 is a slow-moving signal where high frequencies represent sensor jitter.
*   **Resampling:** We upsampled SpO2 from 4 Hz to 32 Hz using SciPy's `resample_poly` to perfectly synchronize it with the respiration arrays without introducing interpolation artifacts.

### Windowing Strategy
The continuous sequences were partitioned into **30-second windows with 50% overlap**.
Labels were extracted from the events file. If an event (Hypopnea or Obstructive Apnea) overlapped the 30-second window bounds by more than 50% (15 seconds), the window took the event's label. Otherwise, it was classified as "Normal".

## 3. Modeling: Multi-Mode Binary Classification 
Due to extreme dataset imbalance (e.g., thousands of "Normal" windows vs. very few Event windows), standard 3-class classification proved unstable. Instead, we architected a `SimpleCNN` taking a `(Batch, 3, 960)` Input Tensor and predicting binary outcomes.

We conducted a Multi-Mode Ablation Study using rigorous **Leave-One-Participant-Out (LOPO)** Cross-Validation to assess true out-of-sample generation:

1.  **Event Mode:** Normal vs. Any Event (Hypopnea + Apnea)
2.  **Hypopnea Mode:** Normal vs. Hypopnea 
3.  **Apnea Mode:** Normal vs. Apnea

## 4. Evaluation and Results

| Mode | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 |
|---|---|---|---|---|
| **Hypopnea** | 0.629 | 0.097 | 0.517 | 0.147 |
| **Event** | 0.555 | 0.116 | 0.590 | 0.172 |
| **Apnea** | 0.919 | 0.019 | 0.187 | 0.033 |

*(Note: The 91.9% accuracy for Apnea is solely due to the model overwhelmingly predicting "Normal", derived from extreme participant imbalance, where over 90% of all Apnea events exist in AP05).*

### Clinical Interpretation & Conclusions
The results confirm physical constraints in the pilot study dataset:

1.  **Strict Imbalance Limitation:** In a LOPO setup, isolated Apnea detection fails because almost all Obstructive Apnea events are contained entirely within a single participant (AP05). The model lacks heterogeneous evidence during training when AP05 is held out.
2.  **Optimal Screening Strategy (Event Merging):** By merging events into a single binary trigger ("Event" Mode), the 1D CNN achieved ~59% Recall. For a medical screening tool trained on only 5 patients, prioritizing high Recall (flagging any irregularity for human review) is the optimal outcome.

## 5. Technical Deliverables
The entire pipeline is robustly encapsulated within modular Python scripts, orchestrated entirely by `AI_for_Health.ipynb` designed for Google Colab environments. The pipeline includes strict per-fold data Normalization to eliminate data-leakage during cross-validation, dynamically applying standard scaling using only the active 4-patient training fold.
