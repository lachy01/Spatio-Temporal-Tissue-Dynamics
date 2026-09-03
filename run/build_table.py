"""Stage 0: build and validate the mixture input table (spec sections 1.1-1.2)."""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, "src")
from tissue_dynamics.mixture.data import build_table, load_config, validate

cfg = load_config()
df = build_table(cfg)
clean, rep = validate(df, cfg)
out = cfg["data"]["table_parquet"]
clean.to_parquet(out, index=False)
os.makedirs("outputs", exist_ok=True)
with open("outputs/data_report.md", "w") as fh:
    fh.write(rep.to_markdown() + "\n")
print("\n" + rep.to_markdown())
print(f"\nwrote {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
