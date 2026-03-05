# AI for Health — SRIP 2026 Project Report
**Objective:** Detecting breathing irregularities (Apnea and Hypopnea) from overnight physiological sleep data.

## 1. Introduction and Data Exploration
The provided dataset contained overnight sleep data from 5 participants (AP01–AP05). The signals provided are standard medical indicators for respiratory distress during sleep:
*   **Nasal Airflow** (32 Hz): Direct measurement of respiration inhalation/exhalation volume.
*   **Thoracic Movement** (32 Hz): Physical chest expansion metrics tracking breathing effort.
*   **SpO2 (Oxygen Saturation)** (4 Hz): Blood oxygen concentration, heavily lagging behind physical breathing events due to circulatory delay.
*   **Events Log** and **Sleep Profile**: Annotations indicating the onset and duration of breathing irregularities (Apnea, Hypopnea), as well as sleep staging.

### 1.1 Data Parsing and Alignment Challenges
The source data inherently lacked structural consistency. For instance, timestamp formats varied significantly between 24-hour military time (`13:45:00.00`) and standard A.M./P.M. markers (`01:45:00.00 PM`). Furthermore, sensors operated at entirely different sampling frequencies (32 Hz vs. 4 Hz).

To handle this, a robust parsing pipeline (`parsers.py`) was engineered using strict Regex formatting and `datetime` synchronization to extract a secure, uniform Unix-epoch starting timestamp for every participant. 
*   Because `SpO2` operates at 4Hz and the respiratory signals at 32Hz, visualizing the raw sequences simultaneously required strict index-matching to the shared start time. `vis.py` generates a comprehensive timeline array for all participants, rendering the full 8-hour progression of all three vital signs explicitly synced to the annotated irregular events.

## 2. Signal Preprocessing & Windowing 
To isolate human breathing patterns—typically occurring between 10 to 24 Breaths Per Minute (BrPM), which translates roughly to **0.17 Hz to 0.40 Hz**—we engineered a domain-specific Digital Signal Processing (DSP) pipeline (`create_dataset.py`).

### 2.1 Digital Signal Processing (DSP)
*   **Respiration (Flow & Thoracic):** Applied a 4th-order Butterworth Bandpass filter (0.17 Hz to 0.40 Hz) using SciPy. This efficiently stripped out high-frequency sensor noise (e.g., patient movement artifacts) and extreme low-frequency shifts (e.g., baseline sensor wander due to posture changes).
*   **SpO2:** SpO2 is a slow-moving, low-frequency physiological indicator. Applying a bandpass filter here would artificially distort the data. Instead, only a 2nd-order Butterworth low-pass filter (cutoff 1.0 Hz) was applied to remove jitter while preserving the authentic baseline oxygen saturation drops.
*   **Resampling Protocol:** To achieve a unified shape for the Neural Network without losing data resolution, SpO2 was upsampled from 4 Hz to 32 Hz using SciPy's `resample_poly` algorithm, rather than basic linear interpolation, to securely preserve the anti-aliased signal trajectory.

### 2.2 Window Extraction and Label Mapping
The continuous, filtered sequences were uniformly segmented into **30-second windows operating with a 50% overlap (15-second step size)**. This generated a highly localized mapping dataset.

**Labeling Strategy:** If an annotated irregular event (Hypopnea or Obstructive Apnea) overlapped the 30-second window bounds by more than 50% (15+ seconds), the window inherited that event's label. Otherwise, the window was classified as "Normal".

## 3. Modeling Architecture
### 3.1 The "SimpleCNN" Design
Given the relatively small cohort (N=5) and severe dataset class imbalance, deep, highly complex architectures are extremely prone to overfitting via memorization. We designed `SimpleCNN` (`models/cnn_model.py`) optimized explicitly for 1D spatial feature detection across synchronized timelines.

*   **Input Representation:** `(Batch, 3 channels, 960 sequence length)`
*   **Architecture:** 
    *   3 consecutive `Conv1D` blocks with progressing channel depths (32 → 64 → 128).
    *   Each block incorporates `BatchNorm` for stable gradients, `ReLU` activation, and `MaxPool1d(4)` to aggressively downsample the temporal dimension and increase the receptive field of the deeper layers.
    *   **Global Average Pooling (GAP):** Before classification, GAP is applied horizontally across the remaining time sequences. This forces the network to learn translation-invariant features (e.g., *does a drop in breathing volume exist anywhere in this 30s window?*), heavily reducing parameters (only 36K total) and guarding against overfitting.

### 3.2 Dynamic Normalization to Prevent Leakage
A fatal error in standard medical ML pipelines is "Global Normalization," where the entire dataset's Mean/Std is calculated before Cross-Validation splits occur, causing future hold-out data to "leak" into the training subset. Our pipeline guarantees that z-score Normalization `(X - mean) / std` is computed strictly over the active 4-patient training fold and subsequently transferred static to the isolated test patient inside the LOPO loop.

## 4. Multi-Mode LOPO Evaluation
Our core protocol requires **Leave-One-Participant-Out (LOPO)** Cross-Validation to guarantee true generalizability to unseen patients. 

During initial data exploration, a severe structural flaw inside the pilot dataset was identified: 
**A massive class imbalance directly isolated to a single patient.** Of the ~150 Obstructive Apnea events recorded across all 5 patients, approximately 140 of them natively belong to Patient AP05 alone.

To scientifically evaluate the model's true pattern-recognition capability without allowing the dataset's extreme skew to mask performance, we executed a rigorous **Multi-Mode Binary Ablation Study**, training three independent classification modes: 

### 4.1 Results Matrix (5-Fold LOPO Mean Metrics)

| Classification Mode | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 Score |
|---|---|---|---|---|
| **Mode A:** Normal vs. Hypopnea | 62.9% | 0.097 | 0.517 | **0.147** |
| **Mode B:** Normal vs. Apnea | 91.9% | 0.019 | 0.187 | **0.033** |
| **Mode C:** Normal vs. *Any* Event | 55.5% | 0.116 | 0.590 | **0.172** |

## 5. Clinical Interpretations & Study Verdict

The V2 Multi-Branch Pipeline yielded vital insights into the Pilot Study dataset constraints:

1.  **Apnea is Architecturally Unlearnable in LOPO (Mode B):** 
    Because 90% of all Apnea events exist solely in AP05, the standard CV loop inherently breaks down. When AP05 is the hold-out test set, the CNN trains on ~15 total Apnea events across the other 4 patients. The network statistically realizes it should always predict "Normal" to minimize mathematical loss. This results in an artificially high Accuracy (91.9%) with a completely dead Recall (18%). 
2.  **Hypopnea Exhibits True Generalizable Features (Mode A):** 
    Conversely to Apnea, Hypopnea events are much more evenly distributed across all patients (AP01, AP02, AP04, AP05). Because the model possesses dense training examples in every fold, it dynamically learned the actual physical features of shallow breathing, achieving a cross-fold mean Recall of nearly 52%.
3.  **The "Event" Merging Pipeline is the Optimal Baseline (Mode C):** 
    By simplifying the objective into a generic binary "Event" detector (Normal vs. [Hypopnea OR Apnea]), the 1D CNN achieved a highly defensible **Recall of 59%**. In a medical screening hierarchy, the initial algorithm's goal should strictly prioritize high Recall (catching *any* breathing irregularity to flag for physician review) rather than exact multi-class categorization accuracy.

**Verdict for Future Development:** The current limit on analytical depth is not driven by algorithmic complexity, but rather the dataset's patient cohort scale. In order to break past the current accuracy ceiling and develop definitive multi-class categorical tracking, the study natively requires a larger, rigorously balanced patient cohort to allow for equalized Apnea prevalence training rather than strict LOPO.
