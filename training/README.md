# Offline defect-classification training

This training track is intentionally separate from the FastAPI app. It is meant to be run in a GPU-enabled Colab or Kaggle notebook so the CodeBERT fine-tuning job can complete efficiently.

## Quick start

1. Create a GPU runtime in Colab or Kaggle.
2. Install the dependencies from [training/requirements.txt](requirements.txt).
3. Run the training script:
   - `python training/train_classifier.py --epochs 3 --batch-size 8 --learning-rate 2e-5 --max-length 256`
4. After training completes, run evaluation:
   - `python training/evaluate.py`

## Notes

- The scripts use the standard train/validation/test splits from the CodeXGLUE Devign dataset.
- Outputs are written to [training/output](output) and [training/metrics.json](metrics.json).
- Expect fine-tuning to take a meaningful amount of time on a GPU, especially for the full dataset and multiple epochs.
