import os
import shutil
import glob

ROOT = r"c:\Users\siddh\Downloads\DACA-HMAS-Dynamic-Adaptive-Communication-Aware-Heterogeneous-Multi-Agent-System-main(1)\DACA-HMAS-Dynamic-Adaptive-Communication-Aware-Heterogeneous-Multi-Agent-System-main"
PLOTS_DIR = os.path.join(ROOT, "plots")
PNG_DIR = os.path.join(PLOTS_DIR, "png")
PDF_DIR = os.path.join(PLOTS_DIR, "pdf")
DATA_DIR = os.path.join(PLOTS_DIR, "data")

ARTIFACTS_DIR = r"C:\Users\siddh\.gemini\antigravity-ide\brain\21541bd7-5e23-4f73-b5e4-74dd38ae90b4"
FIG_SOURCE_DIR = os.path.join(ARTIFACTS_DIR, "figures")

os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Copy PNG and PDF files
png_files = glob.glob(os.path.join(FIG_SOURCE_DIR, "*.png"))
pdf_files = glob.glob(os.path.join(FIG_SOURCE_DIR, "*.pdf"))

print(f"Found {len(png_files)} PNGs and {len(pdf_files)} PDFs.")

for f in png_files:
    shutil.copy(f, PNG_DIR)
    print(f"Copied {os.path.basename(f)} -> plots/png/")

for f in pdf_files:
    shutil.copy(f, PDF_DIR)
    print(f"Copied {os.path.basename(f)} -> plots/pdf/")

# 2. Copy Data Files
stat_csv = os.path.join(ARTIFACTS_DIR, "statistical_summary.csv")
if os.path.exists(stat_csv):
    shutil.copy(stat_csv, DATA_DIR)
    print("Copied statistical_summary.csv -> plots/data/")

agg_json = os.path.join(ROOT, "experiments", "results", "opt1_cqi", "aggregate.json")
if os.path.exists(agg_json):
    shutil.copy(agg_json, DATA_DIR)
    print("Copied aggregate.json -> plots/data/")

all_res_json = os.path.join(ROOT, "experiments", "results", "opt1_cqi", "all_results.json")
if os.path.exists(all_res_json):
    shutil.copy(all_res_json, DATA_DIR)
    print("Copied all_results.json -> plots/data/")

a5_res_json = os.path.join(ROOT, "experiments", "results", "opt1_cqi", "a5_results.json")
if os.path.exists(a5_res_json):
    shutil.copy(a5_res_json, DATA_DIR)
    print("Copied a5_results.json -> plots/data/")

print("Plot organization script complete.")
