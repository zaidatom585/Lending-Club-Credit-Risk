# Databricks notebook source
# MAGIC %md
# MAGIC # 00 Setup
# MAGIC Creates the schema and volumes, then puts each Lending Club file into its own landing folder
# MAGIC (Auto Loader reads a whole folder as a stream). Works with any of these upload methods:
# MAGIC - the Kaggle **zip** uploaded to the `raw` volume (the `.csv.gz` files are extracted from it)
# MAGIC - the **`.csv.gz` (or `.csv`) files** uploaded directly to the `raw` volume (they are moved)
# MAGIC - the files uploaded straight into `landing/accepted` and `landing/rejected` (nothing to do)

# COMMAND ----------

import os
import shutil
import zipfile

CATALOG = "workspace"
SCHEMA = "lending_club"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.checkpoints")

RAW = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
LANDING = {
    "accepted": f"{RAW}/landing/accepted",
    "rejected": f"{RAW}/landing/rejected",
}
for path in LANDING.values():
    os.makedirs(path, exist_ok=True)


def table_for(filename):
    name = filename.lower()
    if name.startswith("accepted"):
        return "accepted"
    if name.startswith("rejected"):
        return "rejected"
    return None


def is_data_file(filename):
    return filename.lower().endswith((".csv.gz", ".csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Extract from a zip, if one was uploaded
# MAGIC The Kaggle zip contains both compressed (`.csv.gz`) and uncompressed copies of each file.
# MAGIC Only the compressed copies are extracted: Spark reads `.csv.gz` directly, and they are much smaller.

# COMMAND ----------

for zname in [f for f in os.listdir(RAW) if f.lower().endswith(".zip")]:
    zip_path = f"{RAW}/{zname}"
    print(f"Found zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        gz = [m for m in members if m.lower().endswith(".csv.gz")]
        chosen = gz if gz else [m for m in members if m.lower().endswith(".csv")]
        for member in chosen:
            name = os.path.basename(member)
            table = table_for(name)
            if table is None:
                print(f"  Skipping {member}")
                continue
            target = f"{LANDING[table]}/{name}"
            if os.path.exists(target):
                print(f"  Already extracted: {target}")
                continue
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
            print(f"  Extracted {member} -> {target}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Move data files uploaded directly to the `raw` volume

# COMMAND ----------

for name in os.listdir(RAW):
    path = f"{RAW}/{name}"
    if os.path.isdir(path) or not is_data_file(name):
        continue
    table = table_for(name)
    if table is None:
        print(f"Not sure which table '{name}' belongs to; leaving it in place.")
        continue
    target = f"{LANDING[table]}/{name}"
    if os.path.exists(target):
        print(f"Already in landing: {target}")
        continue
    shutil.move(path, target)
    print(f"Moved {path} -> {target}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Confirm both landing folders have data

# COMMAND ----------

missing = []
for table, path in LANDING.items():
    files = [f for f in os.listdir(path) if is_data_file(f)]
    if not files:
        missing.append(table)
    for f in files:
        print(f"{table:<9} {os.path.getsize(os.path.join(path, f)) / 1e6:9.1f} MB  {f}")

if missing:
    print("\nFiles currently in the raw volume:")
    for f in sorted(os.listdir(RAW)):
        print("  ", f)
    raise FileNotFoundError(
        f"No data file found for: {missing}. Upload the Kaggle zip, or the accepted/rejected "
        f".csv.gz files, to {RAW}, then run this notebook again."
    )
print("\nBoth landing folders are ready. Run 01_ingest_raw_autoloader next.")