"""Headline kinds and selection shared by the sweep scripts."""
import json

import os
HEAD = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "headlines.json")))
HEAD = HEAD["items"] if isinstance(HEAD, dict) else HEAD
SYM = {"equity": "HLVN", "crypto": "MRD"}
HKINDS = {
    "fake_unv": ("fake", False, "unverified social post", "bearish"),
    "true_ver": ("true", True, "verified wire", "bearish"),
    "fake_ver": ("fake", True, "verified wire", "bearish"),       # e.g. a compromised official account
    "true_unv": ("true", False, "unverified social post", "bearish"),
    "fake_unv_bull": ("fake", False, "unverified social post", "bullish"),
}


def headlines_for(regime, direction, severity, k):
    c = [h for h in HEAD if h["asset"] == SYM[regime] and h["direction"] == direction and h["severity"] in severity]
    return c[:k]
