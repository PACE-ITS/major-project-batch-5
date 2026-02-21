import os
import glob

classes = ['AD', 'CN', 'MCI']
root_raw = os.path.join('data', 'raw')
root_pre = os.path.join('data', 'preprocessed')

total_subjects = 0
results = []
for c in classes:
    adni_dir = os.path.join(root_raw, c, 'ADNI')
    subjects = 0
    if os.path.exists(adni_dir):
        subjects = sum(1 for name in os.listdir(adni_dir) if os.path.isdir(os.path.join(adni_dir, name)))
    samples = len(glob.glob(os.path.join(root_pre, f'{c}_*.npz')))
    total_subjects += subjects
    results.append((c, subjects, samples))

for c, subjects, samples in results:
    print(f"{c}: subjects={subjects}, samples={samples}")
print(f"TOTAL_SUBJECTS={total_subjects}")

# Also print totals for samples
print('TOTAL_SAMPLES=', sum(s for _, _, s in results))
