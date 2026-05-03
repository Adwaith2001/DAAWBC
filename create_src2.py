"""
Run this script from the dynamic_ad_allocation directory to create the src2 structure.
Usage: python create_src2.py
"""

from pathlib import Path

# ======================================================
# ROOT
# ======================================================
ROOT = Path("D:/Research Methodology/DAAWBC/dynamic_ad_allocation")

# ======================================================
# FOLDERS TO CREATE
# ======================================================
folders = [
    # src2 structure
    ROOT / "src2",
    ROOT / "src2" / "simulator",
    ROOT / "src2" / "utils",

    # new ipinyou_v2 data
    ROOT / "data" / "ipinyou_v2",
    ROOT / "data" / "ipinyou_v2" / "1458",
    ROOT / "data" / "ipinyou_v2" / "2259",
    ROOT / "data" / "ipinyou_v2" / "2821",
    ROOT / "data" / "ipinyou_v2" / "2997",
    ROOT / "data" / "ipinyou_v2" / "3358",

    # output folders
    ROOT / "outputs" / "final_experiments_5agents_v3",
    ROOT / "outputs" / "plots_5agents_v3",

    # models folder for future .pt files
    ROOT / "models",
    ROOT / "models" / "5adv_v3",
]

# ======================================================
# EMPTY FILES TO CREATE
# ======================================================
empty_files = [
    ROOT / "src2" / "__init__.py",
    ROOT / "src2" / "simulator" / "__init__.py",
    ROOT / "src2" / "utils" / "__init__.py",
    ROOT / "src2" / "simulator" / "environment_v2.py",
    ROOT / "src2" / "simulator" / "multi_environment_v2.py",
    ROOT / "src2" / "utils" / "pctr_lgbm.py",
    ROOT / "src2" / "policy_network_v2.py",
    ROOT / "src2" / "analyse_data.py",
    ROOT / "src2" / "build_pctr_lgbm.py",
    ROOT / "src2" / "train_v3.py",
    ROOT / "src2" / "plot_v3_results.py",
]

# ======================================================
# CREATE FOLDERS
# ======================================================
print("\n📁 Creating folders...")
for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ {folder}")

# ======================================================
# CREATE EMPTY FILES
# ======================================================
print("\n📄 Creating empty files...")
for f in empty_files:
    if not f.exists():
        f.touch()
        print(f"  ✅ {f}")
    else:
        print(f"  ⚠️  Already exists: {f}")

# ======================================================
# VERIFY STRUCTURE
# ======================================================
print("\n📂 Final src2 structure:")
for path in sorted((ROOT / "src2").rglob("*")):
    depth = len(path.relative_to(ROOT / "src2").parts)
    indent = "  " * depth
    print(f"{indent}{'📁' if path.is_dir() else '📄'} {path.name}")

print("\n📂 Models folder:")
for path in sorted((ROOT / "models").rglob("*")):
    depth = len(path.relative_to(ROOT / "models").parts)
    indent = "  " * depth
    print(f"{indent}{'📁' if path.is_dir() else '📄'} {path.name}")

print("\n✅ Done!")
print(f"📍 src2   : {ROOT / 'src2'}")
print(f"📍 models : {ROOT / 'models'}")