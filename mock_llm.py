from __future__ import annotations

import json
import re

PERIOD_RE = re.compile(r"(\d{4}-\d{2})\s*(?:s/d|sd|-|–|to)\s*(\d{4}-\d{2}|sekarang|present)", re.I)
KEY_RE = re.compile(r"^(NAMA|NAME|LOKASI|LOCATION|KEAHLIAN|SKILLS|SERTIFIKASI|BAHASA|RINGKASAN)\s*:\s*(.*)$", re.I)


def _split(val: str) -> list[str]:
    return [x.strip() for x in re.split(r"[;,]", val) if x.strip() and x.strip() != "-"]


class MockLLM:
    name = "mock-llm (rule-based stand-in)"

    def __init__(self, malformed_first=False, malformed_always=False, hallucinate_skill=None):
        self.malformed_first = malformed_first
        self.malformed_always = malformed_always
        self.hallucinate_skill = hallucinate_skill
        self.calls: list[tuple[str, str, str]] = []

    # ---- public interface (same as real backends) ----
    def complete(self, task: str, system: str, user: str) -> str:
        self.calls.append((task, system, user))
        if task == "extract":
            n_extract = sum(1 for c in self.calls if c[0] == "extract")
            if self.malformed_always or (self.malformed_first and n_extract == 1):
                return 'Sure! Here is the JSON you asked for: {"full_name": '
            cv = re.search(r"<cv>\n(.*?)\n</cv>", user, re.S).group(1)
            p = self._extract(cv)
            if self.hallucinate_skill:  # simulate a model inventing a skill
                p["skills"].append({"name": self.hallucinate_skill,
                                    "evidence": f"Expert in {self.hallucinate_skill} dashboards"})
            return json.dumps(p, ensure_ascii=False)
        return json.dumps(self._recommend(user), ensure_ascii=False)

    # ---- extraction ----
    def _extract(self, cv: str) -> dict:
        p = {"full_name": None, "location": None, "education": [], "work_experience": [],
             "skills": [], "certifications": [], "languages": []}
        section = None
        for raw in cv.splitlines():
            s = raw.strip()
            if not s:
                continue
            m = KEY_RE.match(s)
            if m:
                key, val = m.group(1).upper(), m.group(2).strip()
                section = None
                if key in ("NAMA", "NAME"):
                    p["full_name"] = val or None
                elif key in ("LOKASI", "LOCATION"):
                    p["location"] = val or None
                elif key in ("KEAHLIAN", "SKILLS"):
                    p["skills"] += [{"name": x, "evidence": s} for x in _split(val)]
                elif key == "SERTIFIKASI":
                    p["certifications"] = _split(val)
                elif key == "BAHASA":
                    p["languages"] = _split(val)
                continue  # RINGKASAN is deliberately not extracted
            if s.upper() in ("PENDIDIKAN", "EDUCATION"):
                section = "edu"
                continue
            if s.upper() in ("PENGALAMAN", "EXPERIENCE"):
                section = "exp"
                continue
            if section == "edu" and s.startswith("- "):
                parts = [x.strip() for x in s[2:].split("|")]
                years = re.findall(r"\d{4}", parts[-1]) if len(parts) > 2 else []
                p["education"].append({"institution": parts[0], "degree": parts[1] if len(parts) > 1 else None,
                                       "end_year": int(years[-1]) if years else None, "evidence": s})
            elif section == "exp" and s.startswith("- "):
                parts = [x.strip() for x in s[2:].split("|")]
                per = PERIOD_RE.search(parts[-1]) if len(parts) > 2 else None
                end = None
                if per:
                    end = "present" if per.group(2).lower() in ("sekarang", "present") else per.group(2)
                p["work_experience"].append({
                    "company": parts[0], "title": parts[1] if len(parts) > 1 else "",
                    "start": per.group(1) if per else None, "end": end, "highlights": [], "evidence": s})
            elif section == "exp" and s.startswith("* ") and p["work_experience"]:
                p["work_experience"][-1]["highlights"].append(s[2:])

        work = p["work_experience"]
        fc = {
            "full_name": 0.98 if p["full_name"] else 0.0,
            "location": 0.95 if p["location"] else 0.0,
            "education": 0.95 if p["education"] else 0.0,
            "work_experience": (0.95 if all(w["start"] for w in work) else 0.4) if work else 0.0,
            "skills": 0.95 if p["skills"] else 0.0,
            "certifications": 0.9 if p["certifications"] else 0.0,
            "languages": 0.9 if p["languages"] else 0.0,
        }
        p["field_confidence"] = fc
        p["missing_fields"] = [k for k, v in fc.items() if v == 0.0]
        return p

    # ---- recommendation narrative (works only from the anonymised profile + rubric) ----
    def _recommend(self, user: str) -> dict:
        prof = json.loads(re.search(r"<profile>\n(.*?)\n</profile>", user, re.S).group(1))
        rub = json.loads(re.search(r"<rubric>\n(.*?)\n</rubric>", user, re.S).group(1))
        b = rub["breakdown"]
        strengths, gaps, qs = [], [], []
        if b["must_have"]["matched"]:
            strengths.append("Skill wajib yang terbukti di CV: " + ", ".join(b["must_have"]["matched"]) + ".")
        if b["nice_to_have"]["matched"]:
            strengths.append("Skill tambahan: " + ", ".join(b["nice_to_have"]["matched"]) + ".")
        yrs = prof["total_years_experience"]
        if yrs:
            strengths.append(f"Pengalaman kerja terhitung ±{yrs} tahun berdasarkan tanggal di CV.")
        if b["must_have"]["missing"]:
            gaps.append("Skill wajib belum terlihat di CV: " + ", ".join(b["must_have"]["missing"]) + ".")
            qs.append("Apakah kandidat punya pengalaman " + ", ".join(b["must_have"]["missing"]) +
                      " yang belum tertulis di CV?")
        if yrs is None:
            gaps.append("Durasi pengalaman tidak bisa dihitung.")
            qs.append("Minta kandidat melengkapi periode kerja (bulan/tahun mulai dan selesai).")
        if not prof["education"]:
            gaps.append("Pendidikan tidak tercantum di CV.")
            qs.append("Minta kandidat melengkapi riwayat pendidikan.")
        n_m, n_t = len(b["must_have"]["matched"]), len(b["must_have"]["required"])
        return {
            "strengths": strengths, "gaps": gaps,
            "caveats": ["Skor tidak menilai kualitas pekerjaan sebelumnya, hanya kecocokan terhadap rubrik peran ini."],
            "questions_for_recruiter": qs,
            "rationale": (f"{n_m} dari {n_t} skill wajib terpenuhi dengan skor kecocokan {rub['score']}/100 "
                          f"untuk peran {rub['role']}. Ringkasan ini membantu recruiter memprioritaskan review, "
                          "bukan memutuskan."),
        }
