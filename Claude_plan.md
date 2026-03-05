Good. Let me read every file format before planning anything.I have everything I need. Let me now give you the complete picture before writing a single line of new code.

---

## ⚠️ COMPLETE DATA AUDIT — READ THIS BEFORE ANYTHING ELSE

### Participant → File Mapping (Confirmed)

| Participant | Nasal Airflow | Thoracic | SPO2 | Events | Sleep Profile | Duration |
|---|---|---|---|---|---|---|
| **AP1** | `Flow_Nasal_-_28_05` | `Thorac_Movement_-_28_05` | `SPO2_-_28_05` | `Flow_Events_-_28_05` | `Sleep_profile_-_28_05` | 6.6h |
| **AP2** | `Flow_Signal_-_29_05` | `Thorac_Signal_-_29_05` | `SPO2_Signal_-_29_05` | `Flow_Events_-_29_05` | 🔴 **MISSING** | 8.1h |
| **AP3** | `Flow_-_29_05` | `Thorac_-_29_05` | `SPO2_-_29_05` | 🔴 **MISSING** | `Sleep_profile_-_29_05` | 7.1h |
| **AP4** | `Flow_-_30-05` | `Thorac_-_30-05` | `SPO2_-_30-05` | `Flow_Events_-_30-05` | `Sleep_profile_-_30-05` | 7.6h |
| **AP5** | `Flow__-_30_05` | `Thorac__-_30_05` | `SPO2__-_30_05` | `Flow_Events__-_30_05` | `Sleep_profile__-_30_05` | 7.4h |

**AP2 and AP3 each have one file missing.** This isn't a catastrophe but it's a real constraint — I'll explain how each script handles it.

### Actual File Format (completely different from what I coded before)

**Signal files** (Flow, Thoracic, SPO2): 4-7 header lines → blank → `Data:` → rows of `DD.MM.YYYY HH:MM:SS,mmm; value`

**Two different Start Time formats in headers** (same data, different software export settings):
- Format A: `30-05-2024 21:22:45` (DD-MM-YYYY 24h)
- Format B: `5/28/2024 9:28:07 PM` (M/D/YYYY 12h AM/PM)

**Events files**: `DD.MM.YYYY HH:MM:SS,mmm-HH:MM:SS,mmm; duration_s;EventType; Stage`

**Sleep profile**: `DD.MM.YYYY HH:MM:SS,mmm; StageName` (one epoch every 30s)

**Event label types that actually exist in your data:**
- `Hypopnea` — keep, classify
- `Obstructive Apnea` — keep, classify
- `Mixed Apnea` — keep, classify (only in AP1)
- `Body event` — **discard**, not a breathing irregularity

---

## 🧭 IMPLEMENTATION PLAN

### Step 0 — Data Setup Script (`setup_data.py`)
Copies raw files from wherever they are into the canonical `Data/AP01/` ... `Data/AP05/` folder structure the scripts expect. This is NOT optional — without it, `vis.py -name "Data/AP20"` can't work. This script runs once.

```
Data/
├── AP01/  nasal_airflow.txt, thoracic_movement.txt, spo2.txt, flow_events.txt, sleep_profile.txt
├── AP02/  nasal_airflow.txt, thoracic_movement.txt, spo2.txt, flow_events.txt  [NO sleep_profile]
├── AP03/  nasal_airflow.txt, thoracic_movement.txt, spo2.txt, sleep_profile.txt  [NO flow_events]
├── AP04/  all 5 files
└── AP05/  all 5 files
```

---

### Step 1 — Core Parser (`src/parsers.py`)
This is the foundation everything else builds on. Gets written first, tested in isolation.

**Three parsers:**

**`parse_signal_file(path)`**:
1. Read header lines until `"Data:"` line is found
2. Extract `Start Time` → parse both datetime formats → `recording_start: datetime`
3. Extract `Sample Rate` from header
4. For each data row: parse `DD.MM.YYYY HH:MM:SS,mmm` → `datetime` → subtract `recording_start` → seconds
5. Return `pd.Series(values, index=time_in_seconds)`

**`parse_events_file(path, recording_start: datetime)`**:
1. Skip header lines (no `"Data:"` marker — just skip lines starting with `Signal`, `Start`, `Unit`, `Signal Type`, and blank)
2. For each event line: regex parse `DD.MM.YYYY HH:MM:SS,mmm-HH:MM:SS,mmm; dur; label; stage`
3. Convert start datetime → seconds from recording_start
4. Discard `Body event` rows
5. Return `pd.DataFrame(columns=[start_sec, duration_sec, label, stage])`

**`parse_sleep_profile(path, recording_start: datetime)`**:
1. Skip header, parse `DD.MM.YYYY HH:MM:SS,mmm; Stage` rows
2. Convert timestamps → seconds from recording_start
3. Skip the first `"A"` epoch (artifact marker)
4. Return `pd.DataFrame(columns=[time_sec, stage])`

---

### Step 2 — `vis.py`
Uses the parsers. Generates a 4-panel matplotlib figure saved to PDF.

**Panels**: Nasal Airflow | Thoracic | SPO2 | Sleep Stages

**Key implementation decisions:**
- `sharex=True` across all panels — events align correctly
- X-axis in HH:MM format (seconds → hours:minutes formatter)
- `axvspan()` for event shading — per event type, different colours
- `step()` plot for sleep stages, y-ticks labelled (Wake, REM, N1, N2, N3)
- If sleep profile missing (AP2): panel shows "Not available" text
- Saved via `PdfPages` — single page, landscape A3

**Usage**: `python scripts/vis.py -name "Data/AP01"` → outputs `Visualizations/AP01_visualization.pdf`

---

### Step 3 — `create_dataset.py`
**The most technically critical script.** All ML quality depends on this being correct.

**Pipeline per participant:**
1. Load all 3 signals via `parse_signal_file`
2. Convert SpO2 to absolute timestamps, upsample 4 Hz → 32 Hz via `resample_poly`
3. Align all signals to common time axis: `t_start = max(all signal start times)`, `t_end = min(all signal end times)` — handles slight offsets
4. Apply zero-phase Butterworth filters:
   - Nasal + Thoracic: bandpass 0.17–0.40 Hz (breathing range), order 4, SOS form
   - SpO2: low-pass 0.5 Hz (slow physiological signal)
5. Slide 30s window (960 samples) with 50% overlap (480 step):
   - For each window: compute `>50%` overlap with each event using absolute seconds
   - Label priority if multiple events: highest overlap wins
   - `Body event` already discarded at parse time
   - AP3 has no events → all windows → `Normal` (document this clearly)
6. Store: `participant_id, win_start_s, win_end_s, label` + flattened 960×3 signal values

**Output**: `Dataset/breathing_dataset.csv` + `Dataset/breathing_dataset.pkl`

**Class label mapping** (consolidate for classifier):
- `Normal` → 0
- `Hypopnea` → 1
- `Obstructive Apnea` → 2
- `Mixed Apnea` → 2 (merge with Obstructive — only 2 events in AP1, statistically useless alone)

**Usage**: `python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"`

---

### Step 4 — `models/cnn_model.py`
1D CNN. Architecture stays the same as before — the design was correct. No changes needed.

```
Input (batch, 3, 960)
→ ConvBlock(3→32,  k=7, pool=4) → (batch, 32,  240)
→ ConvBlock(32→64, k=5, pool=4) → (batch, 64,   60)
→ ConvBlock(64→128,k=3, pool=4) → (batch, 128,  15)
→ GlobalAvgPool                  → (batch, 128)
→ Dropout(0.5)
→ Linear(128→num_classes)
```

---

### Step 5 — `train_model.py`
LOPO cross-validation. 5 folds.

**Critical points:**
- Normalisation stats computed on **train fold only**, applied to test — never leaked
- Class weights in `CrossEntropyLoss` — Normal windows will massively outnumber events
- AP3 (no events) = all Normal windows — it will be a hard test participant for the model
- Metrics: Accuracy, Precision, Recall per class, Confusion Matrix per fold + aggregate
- Save: `results/lopo_results.csv`, `results/confusion_matrix_fold{N}.png`, `results/confusion_matrix_overall.png`

**Usage**: `python scripts/train_model.py -dataset_path "Dataset/breathing_dataset.csv"`

---

### Step 6 — `requirements.txt` + `README.md`
Pinned versions, setup instructions, AI disclosure.

---

## Execution Order

```
1. python setup_data.py              ← organise raw files into Data/AP01..AP05
2. python scripts/vis.py -name Data/AP01   ← verify vis works on one participant
3. python scripts/create_dataset.py -in_dir Data -out_dir Dataset
4. python scripts/train_model.py -dataset_path Dataset/breathing_dataset.csv
```

---

## Known Risks to Flag in Your Report

| Risk | Severity | Handling |
|---|---|---|
| AP2 missing sleep profile | Low | vis.py shows "Not available" for that panel |
| AP3 missing events file | **High** | All AP3 windows → Normal. Model sees 0 event examples from this participant. When AP3 is the test fold, test set has only Normal → accuracy will be artificially high for that fold. **Document this.** |
| Only 5 participants | **High** | LOPO with N=5 is extremely high variance. Any single result means nothing; only the aggregate matters. Don't over-interpret individual fold scores. |
| Class imbalance | **High** | Normal >> Hypopnea >> Apnea. Accuracy alone is meaningless. Precision/Recall per class is what matters. |
| `Mixed Apnea` rare | Medium | Only 2 events in AP1. Merged into Apnea class. |

---

Ready to implement? Say **go** and I'll write all scripts in order, starting with the parser since everything depends on it.