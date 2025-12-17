# src/agent_graph.py

from agents.retrieval_agent import RetrievalAgent
from agents.image_agent import ImageAgent
from agents.report_agent import ReportAgent
from agents.reasoner_agent import ReasonerAgent
from agents.verifier_agent import VerifierAgent


class AgentGraph:

    def run(self, query, patient_id, role):
        print("\n================= AGENT GRAPH START =================")

        # Instantiate agents
        retrieval = RetrievalAgent()
        image_agent = ImageAgent()
        report_agent = ReportAgent()
        reasoner = ReasonerAgent()
        verifier = VerifierAgent()

        # ----------------------------------------------------
        # 1️⃣ RETRIEVAL AGENT
        # ----------------------------------------------------
        print("\n[AgentGraph] Step 1 — RetrievalAgent running...")
        evidence = retrieval.run(query, patient_id, role)
        print(f"[AgentGraph] Retrieved {len(evidence)} evidence items.")

        # ----------------------------------------------------
        # 2️⃣ IMAGE CAPTION AGENT
        # ----------------------------------------------------
        print("\n[AgentGraph] Step 2 — ImageAgent generating captions...")
        evidence = image_agent.run(evidence)

        # ----------------------------------------------------
        # 3️⃣ REPORT AGENT (GROUND TRUTH EXTRACTION)
        # ----------------------------------------------------
        print("\n[AgentGraph] Step 3 — ReportAgent extracting ground truth...")
        ground_truth = report_agent.extract_ground_truth(evidence)

        # ----------------------------------------------------
        # 4️⃣ FIRST REASONING PASS (DeepSeek 7B)
        # ----------------------------------------------------
        print("\n[AgentGraph] Step 4 — ReasonerAgent generating answer...")
        answer = reasoner.run(query, evidence)
        print("\n[AgentGraph] Initial Answer Received.")

        # ----------------------------------------------------
        # 5️⃣ VERIFIER LOOP (LangGraph-Style Iterative Correction)
        # ----------------------------------------------------
        print("\n[AgentGraph] Step 5 — Verifier loop started...")

        for round_id in range(1, 4):  # max 3 iterations
            print(f"\n[AgentGraph] --- Verifier Iteration {round_id} ---")

            need_fix, correction = verifier.verify(answer, evidence)

            if not need_fix:
                print("[AgentGraph] Verifier accepted the answer.")
                break

            print("[AgentGraph] Correction required:")
            print(correction)

            # Send correction back to reasoner
            answer = reasoner.run(query, evidence, correction)

        print("\n================= AGENT GRAPH END =================")

        # Return final results to Streamlit
        return {
            "answer": answer,
            "retrieved": evidence,
            "ground_truth": ground_truth
        }
