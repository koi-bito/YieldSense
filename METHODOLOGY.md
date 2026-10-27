# YieldSense — Technical Methodology & Architecture Document

**AI-Enabled NAND Test Yield & Recovery Analytics Pipeline**  
*Authored for SpecTek / Micron MSB Test Solutions Engineering*

---

## 1. Problem Formulation & Industrial Context

In modern semiconductor memory manufacturing (specifically 3D NAND Flash), wafers undergo extensive Automated Test Equipment (ATE) probing across thousands of test vectors—including threshold voltage ($V_t$) distributions, read/program/erase endurance margins, leakage currents, and error correction (ECC) bit-error-rates. 

A non-trivial percentage of rejected dies exhibit marginal failures caused by localized physical process drift rather than catastrophic structural defects. In SpecTek’s business model, recovering and re-binning these marginal dies into suitable lower-tier component specifications represents significant economic value.

YieldSense implements an end-to-end, automated data and machine learning pipeline that:
1. Ingests raw parametric manufacturing test data.
2. Performs data-driven cleaning, outlier imputation, and variance filtering.
3. Conducts statistical process control (SPC) and parametric correlation analysis.
4. Trains an Optuna-optimized XGBoost model with SHAP feature attribution.
5. Scores and ranks failing lots by recovery potential.
6. Serves predictions through a containerized REST API.
7. Produces an automated, rules-validated engineering report via an LLM agent with active hallucination detection.

---

## 2. Dataset & Preprocessing Architecture

### 2.1 Benchmark Selection: UCI SECOM Dataset
The pipeline is demonstrated on the benchmark **UCI SECOM** (Semiconductor Manufacturing) dataset:
- **Instances**: 1,567 manufacturing lots / wafers.
- **Features**: 590 continuous sensor channels capturing in-situ chamber pressures, temperatures, RF power, gas flows, and parametric electrical readings.
- **Target**: Binary classification (0 = Pass, 1 = Fail).
- **Class Imbalance**: 1,463 passing lots (93.36%) vs. 104 failing lots (6.64%).

### 2.2 Preprocessing Decisions & Engineering Rationales

| Decision | Implementation | Engineering Rationale |
|---|---|---|
| **High Missingness Removal** | Dropped features with $>40\%$ nulls (32 features dropped) | Features with extreme missingness introduce severe synthetic noise when imputed, obscuring authentic physical drift signals. |
| **Imputation Strategy** | Column Median Imputation | Semiconductor test readings are prone to high-sigma electrical measurement spikes. Median is non-parametric and robust to extreme outliers, whereas mean imputation severely shifts the center line. |
| **Zero-Variance Filtering** | Dropped features with variance $\le 0.0$ (116 features dropped) | Invariant sensor channels convey zero informational entropy and increase matrix conditioning penalties without contributing discriminative power. |
| **Active Feature Retention** | 442 clean sensor channels | Preserves maximum parametric resolution while eliminating 148 dead channels. |
| **Leak-Free Scaling** | `StandardScaler` fitted strictly on train split | Scaling before splitting introduces test-set distribution leakage into the training phase. The scaler is fit only on the 80% train split and serialized to `models/scaler.pkl` for inference. |

---

## 3. Statistical Analysis Methodology

### 3.1 Point-Biserial Parametric Correlation
Point-biserial correlation coefficients ($r_{pb}$) and corresponding two-tailed $p$-values are computed for all 442 sensors against the binary pass/fail target:
$$r_{pb} = \frac{\bar{X}_1 - \bar{X}_0}{s_X} \sqrt{\frac{n_1 n_0}{n(n-1)}}$$
- **Findings**: 86 sensor parameters display statistically significant correlation ($p < 0.05$).
- **Top Drivers**: `sensor_060` ($r = 0.155$), `sensor_104` ($r = 0.144$), and `sensor_511` ($r = 0.141$) represent primary physical indicators of wafer probe failure.

### 3.2 Principal Component Analysis (PCA)
To evaluate sensor collinearity, standardized sensor readings were decomposed via PCA:
- **Results**: 161 Principal Components explain $95.1\%$ of cumulative sensor variance (a $63.6\%$ dimensional reduction from 442 sensors).
- **Interpretation**: Strong physical coupling exists across multi-sensor cluster tools (e.g., thermal and gas flow interactions in deposition chambers).

### 3.3 Statistical Process Control (SPC) Shewhart 3-Sigma Control Charts
For the top failure-correlated sensors, chronological moving averages and 3-sigma control limits ($UCL = \bar{X} + 3\sigma$, $LCL = \bar{X} - 3\sigma$) were calculated over a 30-lot moving window:
- **Drift Events**: 208 out-of-control excursions were flagged across the top 10 monitored channels.
- **Physical Meaning**: Identifies early equipment wear and chamber degradation prior to catastrophic wafer scrap events.

---

## 4. Machine Learning & Recovery Scoring

### 4.1 Imbalance Mitigation: Leak-Free SMOTE
Due to severe 93.4% / 6.6% class imbalance, Synthetic Minority Over-sampling Technique (SMOTE) was applied **strictly within the training fold** after the 80/20 stratified split. This synthesizes minority class examples along feature space line segments connecting $k$-nearest neighbors without contaminating the holdout test set.

### 4.2 Model Optimization & Optuna Tuning
An XGBoost gradient boosted decision tree classifier was selected as the primary estimator due to its handling of non-linear interactions, tolerance to correlated inputs, and compatibility with exact TreeSHAP attribution.

Hyperparameter optimization was conducted using **Optuna** over 25 trials with 3-fold stratified cross-validation optimizing ROC-AUC:
- `n_estimators`: 100
- `max_depth`: 4
- `learning_rate`: 0.052
- `subsample`: 0.84
- `colsample_bytree`: 0.72
- `scale_pos_weight`: 2.85

### 4.3 Evaluation Metrics

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.5210 | 0.0712 | 0.0811 | 0.1428 | 0.1034 |
| **XGBoost (Optuna Tuned)** | **0.6442** | **0.1574** | **0.2500** | **0.0476** | **0.0800** |

*Note: In semiconductor yield analytics, ROC-AUC and continuous failure probability calibration are the key engineering metrics because lots are triaged by risk and proximity rather than hard thresholding.*

### 4.4 Explainable AI (XAI) via TreeSHAP
Using Lundberg's `TreeExplainer`, Shapley values are calculated for each test lot:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
Where $\phi_i(x)$ quantifies the exact directional push (positive = fail risk, negative = pass affinity) of sensor $i$. This eliminates the black-box nature of tree ensembles, providing process engineers with the exact top 5 root-cause sensors per defective wafer.

### 4.5 Recovery Scoring Formulation
A composite recovery score $S_{\text{rec}} \in [0.0, 1.0]$ ranks failing lots by salvageability:
$$S_{\text{rec}} = 0.60 \cdot \left(1.0 - \frac{P(\text{Fail}) - 0.50}{0.50}\right) + 0.30 \cdot \left(1.0 - \frac{\min(K, K_{\max})}{K_{\max}}\right) + 0.10 \cdot B_{\text{rule}}$$
Where:
- $P(\text{Fail}) \in [0.50, 1.00]$ is the predicted failure probability (boundary proximity).
- $K$ is the number of adverse driving sensors (defect sparsity).
- $B_{\text{rule}} \in \{0, 1\}$ is a bonus indicator if the lot satisfies physical 3-sigma recovery boundaries.

**Yield Recovery Impact**: Triaging identified **33 recovery candidate lots (30.8% of defect units)** displaying marginal failure profiles suitable for component downgrading.

---

## 5. Automated AI Reporting & Governance Layer

### 5.1 Architecture
The reporting agent synthesizes analytical outputs into an executive-ready Markdown document. To prevent LLM hallucinations, a deterministic **Validation Layer** audits all claims before publication:

```
[Pipeline Artifacts (CSVs / JSONs)]
                │
                ▼
      [Context Assembler]  ───>  JSON Context
                │
                ▼
      [LLM Drafter / Mock] ───>  Draft Report
                │
                ▼
    [Rules-Based Validator] ───>  Hallucination Check & Number Grounding
                │
                ▼
    [Validated Engineering Report Artifact]
```

### 5.2 Verification Guardrails
1. **Sensor Identifier Whitelist**: Audits every referenced sensor against the 442 active features in DuckDB. Flags fictitious citations (e.g., `sensor_999`) as critical failures (`FAILED`).
2. **Numerical Tolerance Verification**: Audits every metric, count, and percentage against ground-truth context numbers within a $2\%$ relative tolerance.
3. **Disclosure Header**: Formats a transparent audit table directly into the generated report artifact showing claim count, verification rate, and discrepancy alerts.

---

## 6. Engineering Limitations & Production Roadmap

1. **Sensor Anonymization**: The benchmark dataset anonymizes sensor physical definitions. In production at SpecTek, features will map directly to equipment chamber IDs, tool recipes, and ATE test bins.
2. **Concept Drift**: Semiconductor process nodes experience tool seasoning, target adjustments, and preventive maintenance shifts. An automated cron retraining pipeline should re-tune parameters monthly.
3. **Integration**: The FastAPI service provides `POST /analyze-lot` and `POST /batch-analyze` endpoints ready for containerized deployment in Kubernetes adjacent to factory floor MES/ATE systems.
4. **Computational Vectorization Tradeoff**: Current implementation is optimized for the SECOM dataset scale; vectorization via NumPy broadcasting is a documented next step for larger datasets (e.g., replacing row-by-row iteration in recovery candidate evaluation and SPC drift event collection with vectorized tensor operations for fab-scale throughput).
