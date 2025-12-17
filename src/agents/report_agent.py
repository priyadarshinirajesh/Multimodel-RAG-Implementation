# src/agents/report_agent.py

class ReportAgent:
    def extract_ground_truth(self, evidence):
        truth = ""

        for e in evidence:
            if e["source"] == "patient_record" and e.get("impression"):
                truth += e["impression"] + " "

        return truth.strip()
