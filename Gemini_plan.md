1. ⚠️ REVIEW

Stopping to plan before writing more code is the first solid engineering decision you've made today. Diving straight into modeling without a blueprint for multimodal time-series alignment is exactly how you end up with data leakage and garbage predictions.

I will lay out the entire end-to-end architecture and implementation plan below.

**Disclaimer check:** The assignment instructions explicitly state that using AI tools like me requires full disclosure in your submission. You also must be able to explain everything we build here during your 1-on-1 discussion. If you try to pass this off as solely your own work without understanding it, they will know, and your chances drop to zero. I am proceeding under the assumption that you will disclose this and study the code I provide.

2. 🧭 APPROACH — PHASE 0 PROJECT PLAN

Here is your master blueprint. We are treating this as a **Supervised Time-Series Window Classification** problem.

### 1. Problem Definition & Metrics

* 
**Formulation:** Given 30-second multi-channel signal windows (Airflow, Thoracic, SpO2), predict the correct breathing state.


* **Output Type:** Multi-class classification. The labels will be "Normal" plus whatever specific events are in your `flow_events.csv` (e.g., "Hypopnea", "Obstructive Apnea").


* 
**Target Metric:** You must report Accuracy, Precision, Recall, and a Confusion Matrix. However, sleep data is heavily skewed towards "Normal". We will optimize for **Macro F1-Score** during training so the model is penalized for ignoring the rare apnea events.


* **Baseline Model:** A naive "Majority Class Predictor" (predicting "Normal" for every window). If our 1D CNN cannot beat this naive baseline in F1-score, the model is failing.

### 2. Implementation Pipeline

| Phase | Script | Objective | Key Engineering Tasks |
| --- | --- | --- | --- |
| **Phase 1: Setup & EDA** | `vis.py` | Visualize 8-hour signals + events.

 | • Map raw files to `Data/APXX/` structure .

<br>

<br>• Safely parse messy `.txt` delimiters.<br>

<br>• Plot 32Hz/4Hz signals on a shared timeline .

<br>

<br>• Overlay annotations and save to lightweight PDF.

 |
| **Phase 2: Signal Filtering** | `create_dataset.py` | Remove noise outside human breathing frequencies.

 | • Apply a digital Bandpass Filter (0.17 Hz to 0.4 Hz) using `scipy.signal`.

<br>

<br>• Filter both 32Hz and 4Hz signals appropriately. |
| **Phase 3: Windowing & Labeling** | `create_dataset.py` | Chunk 8-hour data into model inputs.

 | • Slice signals into 30-second windows with 50% overlap .

<br>

<br>• **Labeling Logic:** If an event overlaps >50% of the window, assign that event's label. Else, label "Normal" .

<br>

<br>• Save arrays as PyTorch-ready `.pkl` or `.npy`.

 |
| **Phase 4: Model Architecture** | `cnn_model.py` | Build the PyTorch network.

 | • 1D CNN with multi-branch inputs (to handle 32Hz vs 4Hz dimensionality differences).<br>

<br>• Global Average Pooling to prevent overfitting.<br>

<br>• Fully connected classification head. |
| **Phase 5: LOOCV Training** | `train_model.py` | Train robustly without data leakage.

 | • **Loop 5 times:** Train on 4 subjects, Test on 1 subject.

<br>

<br>• Handle class imbalance using `WeightedRandomSampler` or Class Weights in CrossEntropyLoss.<br>

<br>• Track metrics per fold. |
| **Phase 6: Final Eval & Packaging** | `README.md` | Aggregate results and format repo.

 | • Calculate final CV metrics (Accuracy, Precision, Recall) .

<br>

<br>• Generate global confusion matrix .

<br>

<br>• Ensure repo matches the exact required structure.

 |

3. 💻 CODE

*(No code is required for the planning phase. We will resume coding once the blockers below are resolved).*

4. 🔍 EDGE CASES (Failure Points We Must Watch)

* 
Temporal Misalignment : The SpO2 is sampled at 4 Hz, while Airflow is 32 Hz. A 30-second window means exactly 960 points for Airflow/Thoracic, but only 120 points for SpO2. We cannot just stack these into a single matrix. We either need to upsample SpO2 to 32 Hz using interpolation, or use a multi-branch CNN that processes them separately before concatenating the features. **(I recommend interpolation for simplicity in a baseline CNN).**


* **Cross-Validation Leakage:** If we standardize/scale the signals using the *entire* dataset before splitting, we leak future data into the training set. We must normalize the 4 training participants independently of the 1 test participant inside the LOOCV loop.


* **The Midnight Rollover:** If your timestamps are stored as `23:59:59` transitioning to `00:00:00`, simple math will think time went backward. We have to parse these as proper Pandas `datetime` objects or continuous seconds.

5. ➡️ NEXT STEP

The plan is set. To actually execute **Phase 1** and write the correct parsing logic for `vis.py` and `create_dataset.py`, I still need to know exactly what is inside your files.

**Do this right now:**

1. Open one `nasal_airflow.txt` file and paste the first 3 lines.
2. Open one `flow_events.txt` (or `.csv`) file and paste the first 3 lines.

I need to see the column headers, the delimiter (commas or tabs?), and the exact timestamp format. Once you paste those, we immediately write the final, working version of `vis.py`.