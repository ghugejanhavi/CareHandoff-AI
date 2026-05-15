"""
MIMIC-IV Data Loader
Handles loading discharge notes (MIMIC-IV-Note) and structured EHR tables
(prescriptions, lab events, diagnoses, procedures, admissions).

Place your MIMIC-IV files in:  data/raw/mimic/
Supported formats: .csv  or  .csv.gz  (auto-detected)
"""

import os
import pandas as pd
from typing import Dict, List, Optional
import logging

from config import MIMIC_DIR, MIMIC_FILES

logger = logging.getLogger(__name__)


class MIMICLoader:
    """
    Loads MIMIC-IV CSV files and provides per-admission EHR snapshots.
    All heavy DataFrames are lazily loaded and cached after first access.
    """

    def __init__(self, data_dir: str = MIMIC_DIR):
        self.data_dir = data_dir
        self._cache: Dict[str, pd.DataFrame] = {}

    # ── Internal helpers ───────────────────────────────────────────────────

    def _resolve_path(self, key: str) -> Optional[str]:
        """Return path to a MIMIC file, trying .csv then .csv.gz."""
        base = MIMIC_FILES[key]
        for name in (base, base + ".gz"):
            path = os.path.join(self.data_dir, name)
            if os.path.exists(path):
                return path
        return None

    def _load(self, key: str, **kwargs) -> pd.DataFrame:
        """Load a MIMIC CSV (cached)."""
        if key in self._cache:
            return self._cache[key]

        path = self._resolve_path(key)
        if path is None:
            logger.warning("MIMIC file not found for key '%s' in %s", key, self.data_dir)
            return pd.DataFrame()

        logger.info("Loading MIMIC table '%s' from %s", key, path)
        df = pd.read_csv(path, low_memory=False, **kwargs)
        self._cache[key] = df
        return df

    # ── Public loaders ─────────────────────────────────────────────────────

    def load_discharge_notes(self) -> pd.DataFrame:
        """
        Returns DataFrame with columns: subject_id, hadm_id, text, charttime.
        Source: MIMIC-IV-Note  discharge.csv
        """
        df = self._load("discharge")
        if df.empty:
            return df
        keep = [c for c in ["subject_id", "hadm_id", "note_id", "charttime", "text"] if c in df.columns]
        return df[keep].dropna(subset=["hadm_id", "text"])

    def load_prescriptions(self) -> pd.DataFrame:
        """
        Returns medication orders with: hadm_id, drug, dose_val_rx, route, starttime, stoptime.
        """
        df = self._load("prescriptions")
        if df.empty:
            return df
        keep = [c for c in ["hadm_id", "drug", "dose_val_rx", "dose_unit_rx", "route", "starttime", "stoptime"] if c in df.columns]
        return df[keep].dropna(subset=["hadm_id", "drug"])

    def load_labevents(self) -> pd.DataFrame:
        """
        Returns lab results joined with item labels: hadm_id, label, value, flag, charttime.
        """
        labs = self._load("labevents")
        items = self._load("d_labitems")
        if labs.empty:
            return labs
        if not items.empty and "itemid" in labs.columns and "itemid" in items.columns:
            labs = labs.merge(items[["itemid", "label"]], on="itemid", how="left")
        keep = [c for c in ["hadm_id", "itemid", "label", "value", "valuenum", "flag", "charttime"] if c in labs.columns]
        return labs[keep].dropna(subset=["hadm_id"])

    def load_diagnoses(self) -> pd.DataFrame:
        """
        Returns diagnoses with ICD descriptions: hadm_id, icd_code, icd_version, long_title.
        """
        diag = self._load("diagnoses")
        desc = self._load("d_icd_diagnoses")
        if diag.empty:
            return diag
        if not desc.empty:
            on_cols = [c for c in ["icd_code", "icd_version"] if c in diag.columns and c in desc.columns]
            if on_cols:
                diag = diag.merge(desc[on_cols + ["long_title"]], on=on_cols, how="left")
        keep = [c for c in ["hadm_id", "icd_code", "icd_version", "long_title", "seq_num"] if c in diag.columns]
        return diag[keep].dropna(subset=["hadm_id"])

    def load_procedures(self) -> pd.DataFrame:
        """Returns procedures: hadm_id, icd_code, icd_version, long_title."""
        proc = self._load("procedures")
        desc = self._load("d_icd_procedures")
        if proc.empty:
            return proc
        if not desc.empty:
            on_cols = [c for c in ["icd_code", "icd_version"] if c in proc.columns and c in desc.columns]
            if on_cols:
                proc = proc.merge(desc[on_cols + ["long_title"]], on=on_cols, how="left")
        keep = [c for c in ["hadm_id", "icd_code", "icd_version", "long_title"] if c in proc.columns]
        return proc[keep].dropna(subset=["hadm_id"])

    def load_admissions(self) -> pd.DataFrame:
        """Returns admissions: hadm_id, subject_id, admittime, dischtime, discharge_location."""
        df = self._load("admissions")
        if df.empty:
            return df
        keep = [c for c in ["hadm_id", "subject_id", "admittime", "dischtime", "discharge_location", "admission_type"] if c in df.columns]
        return df[keep].dropna(subset=["hadm_id"])

    # ── Composite EHR snapshot ─────────────────────────────────────────────

    def get_patient_ehr(self, hadm_id: int) -> Dict:
        """
        Return a structured EHR snapshot for one admission.
        Used by agents for comparison against the discharge note.
        """
        hadm_id = int(hadm_id)

        prescriptions = self.load_prescriptions()
        labevents = self.load_labevents()
        diagnoses = self.load_diagnoses()
        procedures = self.load_procedures()
        admissions = self.load_admissions()

        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or "hadm_id" not in df.columns:
                return pd.DataFrame()
            return df[df["hadm_id"] == hadm_id]

        meds_df = _filter(prescriptions)
        labs_df = _filter(labevents)
        diag_df = _filter(diagnoses)
        proc_df = _filter(procedures)
        adm_df = _filter(admissions)

        # Pending labs: rows where no result value has been recorded yet
        pending_labs = []
        if not labs_df.empty and "value" in labs_df.columns:
            pending_mask = labs_df["value"].isna()
            pending_labs = labs_df[pending_mask]["label"].dropna().unique().tolist()

        return {
            "hadm_id": hadm_id,
            "admission": adm_df.to_dict("records")[0] if not adm_df.empty else {},
            "diagnoses": diag_df["long_title"].dropna().tolist() if "long_title" in diag_df.columns else [],
            "medications": meds_df["drug"].dropna().unique().tolist() if "drug" in meds_df.columns else [],
            "medications_detail": meds_df.to_dict("records") if not meds_df.empty else [],
            "lab_results": labs_df.to_dict("records") if not labs_df.empty else [],
            "pending_labs": pending_labs,
            "procedures": proc_df["long_title"].dropna().tolist() if "long_title" in proc_df.columns else [],
        }

    def list_available_admissions(self, n: int = 20) -> List[int]:
        """Return a sample of hadm_ids that have discharge notes."""
        notes = self.load_discharge_notes()
        if notes.empty or "hadm_id" not in notes.columns:
            return []
        return notes["hadm_id"].dropna().astype(int).unique().tolist()[:n]

    def is_data_available(self) -> Dict[str, bool]:
        """Check which MIMIC files are present."""
        return {key: self._resolve_path(key) is not None for key in MIMIC_FILES}
