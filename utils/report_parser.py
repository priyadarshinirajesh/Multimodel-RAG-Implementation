# utils/report_parser.py

import re

def split_report_sections(report_text: str) -> dict:
    txt = report_text or ""
    low = txt.lower()

    def _extract(start_key, end_keys):
        s = low.find(start_key)
        if s == -1:
            return ""
        s2 = s + len(start_key)
        e = len(txt)
        for k in end_keys:
            i = low.find(k, s2)
            if i != -1:
                e = min(e, i)
        return txt[s2:e].strip(" :\n\t")

    return {
        "indication": _extract("indication:", ["comparison:", "findings:", "impression:"]),
        "comparison": _extract("comparison:", ["findings:", "impression:"]),
        "findings": _extract("findings:", ["impression:"]),
        "impression": _extract("impression:", []),
    }
