# Lab Assignment 3 — Submission Checklist

## What Needs to Be Submitted

| Item | Where | Status |
|------|--------|--------|
| PDF (report + code appendix) | **Gradescope** | ⚠️ Need to compile .tex → .pdf |
| `raj_kanakarajan_challenge1.pkl` | **Canvas** | ✅ Ready |
| `raj_kanakarajan_challenge2.json` | **Canvas** | ✅ Ready |
| `raj_kanakarajan_challenge3.pkl` | **Canvas** | ✅ Ready |
| `raj_kanakarajan_challenge4.json` | **Canvas** | ✅ Ready |

---

## ⚠️ Critical Issues Before Submitting

### 1. Data Efficiency Results Are Not Real (5922 Required Section)
**File:** `Lab Assignment 3.tex`, Section 3 (Data Efficiency), Table `tab:data_eff`

The 75% and 100% training data results use `\sim` (approximation):
```
End-to-End:  50% = 74.3  |  75% = ~76.1  |  100% = ~77.9
CLIP-based:  50% = 81.5  |  75% = ~82.3  |  100% = ~83.0
```

The 50% values are real (from `experiment_results.json`). The 75% and 100% values are **estimated**, not experimentally obtained. Your `theory_and_report.md` notes explicitly say:
> *"These are estimates based on the reasoning above; actual results to be filled in."*

**What to do:**
- Run `run_experiments.py` (or write a new script) to actually train both binary classification models (E2E + CLIP) at 75% and 100% data fractions
- Replace `~76.1`, `~77.9`, `~82.3`, `~83.0` in the LaTeX with actual numbers
- Remove the `\sim` prefixes once real values are in

---

### 2. Compile LaTeX to PDF
The report lives in `Lab Assignment 3.tex`. You need a PDF to submit to Gradescope.

**Options:**
- **Overleaf** (recommended): Create a new project, upload `Lab Assignment 3.tex` + `run_e2e.py` + `solution_clip.py` (the appendix uses `\lstinputlisting` to pull them in), then download the compiled PDF.
- **Local LaTeX**: Run `pdflatex "Lab Assignment 3.tex"` (requires `pdflatex` installed).

**Note:** The appendix uses:
```latex
\lstinputlisting[language=Python]{run_e2e.py}
\lstinputlisting[language=Python]{solution_clip.py}
```
Both files exist in the directory, so compilation should work.

---

### 3. Consider Including `run_experiments.py` in the Appendix
The appendix currently includes `run_e2e.py` and `solution_clip.py`, but `run_experiments.py` contains the actual code that produced the Exp 4.1 / 4.2 results. The assignment says "all your code must be copy-pasted into the appendix."

**Options:**
- Add `\lstinputlisting[language=Python]{run_experiments.py}` after `solution_clip.py` in the appendix
- Or, since `run_experiments.py` is a subset of what's already in `solution_clip.py`, you may judge it as redundant. However, to be safe, include it.

---

## ✅ What Is Already Correct

### Prediction Files (all verified)
| File | Samples | Format | Test Indices |
|------|---------|--------|--------------|
| `raj_kanakarajan_challenge1.pkl` | 100 | 1D int64 tensor, values {0,1} | 100–199 ✅ |
| `raj_kanakarajan_challenge2.json` | 100 | `[{"image": ..., "answer": ...}]` | 100–199 ✅ |
| `raj_kanakarajan_challenge3.pkl` | 100 | 1D int64 tensor, values {0,1} | 100–199 ✅ |
| `raj_kanakarajan_challenge4.json` | 100 | `[{"image": ..., "answer": ...}]` | 100–199 ✅ |

### Report Structure
- ✅ Methods / Results / Analysis structure followed in all 3 sections
- ✅ Preprocessing strategy described (Section 1 Methods)
- ✅ At least 1 controlled experiment per task for E2E (Exp 3.1: nhead ablation, Exp 3.2: supervision strategy)
- ✅ At least 1 controlled experiment per task for CLIP (Exp 4.1: fusion strategy, Exp 4.2: vocab size)
- ✅ Final architecture descriptions included for all 4 challenges
- ✅ CLIP vs. E2E comparison included (Section 2 Analysis)
- ✅ Data efficiency hypothesis written (Section 3 — required before running experiments)
- ✅ Data efficiency methods described (hyperparameters, data splits, seed)
- ✅ Code in appendix as plain text via `\lstinputlisting` (not screenshots)
- ✅ 50% training data cap respected (first 10,000 of 20,523 samples)
- ✅ Both architectures use visual + textual inputs
- ✅ E2E architectures use self/cross-attention (TransformerEncoderLayer + FusionAttn)
- ✅ No pre-trained features/architectures used for E2E models
- ✅ File names follow exact format: `firstname_lastname_challenge{N}.(pkl|json)`

---

## Final Submission Steps (In Order)

1. **Run data efficiency experiments** — train E2E and CLIP classifiers at 75% and 100% data; note actual val accuracy for each
2. **Update the LaTeX table** — replace `$\sim$76.1` etc. with real values; update the analysis paragraph if results differ significantly from what's written
3. **Compile the PDF** — use Overleaf with all three files: `.tex`, `run_e2e.py`, `solution_clip.py`
4. **Submit PDF to Gradescope**
5. **Submit 4 prediction files to Canvas**
