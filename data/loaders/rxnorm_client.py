"""
RxNorm API Client (NLM)
Used to normalize drug names before EHR-vs-note comparison
so that "Metoprolol Succinate" and "metoprolol succinate ER" map to the same RxCUI.

No API key required. Base URL: https://rxnav.nlm.nih.gov/REST
"""

import logging
import requests
from typing import Optional, List, Dict
from functools import lru_cache

from config import RXNORM_BASE_URL, RXNORM_TIMEOUT_SEC

logger = logging.getLogger(__name__)


class RxNormClient:
    """Lightweight wrapper around the NLM RxNorm REST API."""

    def __init__(self, base_url: str = RXNORM_BASE_URL, timeout: int = RXNORM_TIMEOUT_SEC):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ── Core lookup ────────────────────────────────────────────────────────

    @lru_cache(maxsize=512)
    def get_rxcui(self, drug_name: str) -> Optional[str]:
        """
        Return the RxCUI (concept identifier) for a drug name.
        Returns None if not found or on network error.
        """
        try:
            url = f"{self.base_url}/rxcui.json"
            resp = self._session.get(url, params={"name": drug_name}, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            rxcui = data.get("idGroup", {}).get("rxnormId", [])
            return rxcui[0] if rxcui else None
        except Exception as e:
            logger.debug("RxNorm lookup failed for '%s': %s", drug_name, e)
            return None

    @lru_cache(maxsize=512)
    def normalize_name(self, drug_name: str) -> str:
        """
        Return the canonical RxNorm name for a drug.
        Falls back to the original name if not found.
        """
        rxcui = self.get_rxcui(drug_name)
        if not rxcui:
            return drug_name.lower().strip()
        try:
            url = f"{self.base_url}/rxcui/{rxcui}/properties.json"
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            name = resp.json().get("properties", {}).get("name", drug_name)
            return name.lower().strip()
        except Exception:
            return drug_name.lower().strip()

    # ── Batch normalization ────────────────────────────────────────────────

    def normalize_drug_list(self, drug_names: List[str]) -> Dict[str, str]:
        """
        Normalize a list of drug names.
        Returns {original_name: normalized_name}.
        Network errors are silently swallowed; original name is preserved.
        """
        return {name: self.normalize_name(name) for name in drug_names}

    # ── Drug interaction check ─────────────────────────────────────────────

    def check_interactions(self, rxcuis: List[str]) -> List[Dict]:
        """
        Return drug-drug interaction data for a list of RxCUIs.
        Returns empty list on error or if no interactions found.
        """
        if len(rxcuis) < 2:
            return []
        try:
            url = f"{self.base_url}/interaction/list.json"
            resp = self._session.get(url, params={"rxcuis": "+".join(rxcuis)}, timeout=self.timeout)
            resp.raise_for_status()
            pairs = resp.json().get("fullInteractionTypeGroup", [])
            interactions = []
            for group in pairs:
                for itype in group.get("fullInteractionType", []):
                    for pair in itype.get("interactionPair", []):
                        interactions.append({
                            "description": pair.get("description", ""),
                            "severity": pair.get("severity", "unknown"),
                            "drugs": [c.get("minConceptItem", {}).get("name", "") for c in pair.get("interactionConcept", [])],
                        })
            return interactions
        except Exception as e:
            logger.debug("Interaction check failed: %s", e)
            return []

    def is_available(self) -> bool:
        """Ping the API to check connectivity."""
        try:
            resp = self._session.get(f"{self.base_url}/version.json", timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False
