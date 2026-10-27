# SECOM Dataset Documentation

## Source

**UCI Machine Learning Repository — SECOM Dataset**
- URL: https://archive.ics.uci.edu/dataset/179/secom
- Donated by: Michael McCann, Adrian Johnston (2008)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Description

The SECOM dataset contains sensor measurements from a semiconductor manufacturing process. Each row represents a single production unit (wafer/lot) passing through a test flow, with sensor readings captured at various stages.

## Files

| File | Format | Description |
|------|--------|-------------|
| `secom.data` | Space-delimited, no header | 590 sensor readings per row, 1567 rows |
| `secom_labels.data` | Space-delimited | Column 1: label (-1 = pass, 1 = fail), Column 2: timestamp |

## Column Semantics

- **Sensors 1–590**: Anonymized process and test parameters (temperature, voltage, current, pressure, flow rates, etc.). Column names are not provided — we assign `sensor_001` through `sensor_590`.
- **Label**: Binary classification target. `-1` = pass (functional unit), `1` = fail (defective unit).
- **Timestamp**: Date-time of the test event (format: `DD/MM/YYYY HH:MM:SS`).

## Known Quality Issues

These are intentional and reflect real manufacturing data characteristics:

1. **High missingness**: Some sensor columns have >40% missing values (sensor offline, read errors, or not applicable to certain product types)
2. **Severe class imbalance**: ~93.4% pass (-1), ~6.6% fail (1) — typical for a mature manufacturing process
3. **Near-zero variance features**: Several sensors report nearly constant values and carry no discriminative signal
4. **No feature descriptions**: Sensor identities are anonymized — analysis must be purely data-driven
5. **Potential multicollinearity**: Many sensors measure correlated physical phenomena

## Mapping to NAND Test-Yield Domain

| SECOM Concept | NAND Test-Flow Equivalent |
|---------------|--------------------------|
| Wafer/unit | NAND die or lot under test |
| 590 sensor readings | Parametric test measurements (Vt distributions, read/program/erase voltages, bit error rates) |
| Pass/fail label | Bin classification (good die vs reject) |
| Recovery candidates | Failing die that could be re-binned to a lower-grade product (SpecTek's core business) |
| Timestamp | Lot processing timestamp for SPC trending |

## Citation

```
McCann, M. & Johnston, A. (2008). SECOM [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5K89R
```
