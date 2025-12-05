from src.rbac import apply_rbac
from src.retrieve import retrieve_multimodal
from src.rag_answer import make_rag_answer

def chat():
    print("🔵 Multimodal RAG Medical Assistant")
    query = input("Enter your medical query: ")
    role = input("Enter role (doctor/nurse/patient/admin): ")

    # Step 1: Retrieve from database
    retrieved = retrieve_multimodal(query, top_k=5)

    # Step 2: Apply RBAC filtering
    filtered = apply_rbac(role, retrieved)

    # Step 3: Generate RAG answer
    answer = make_rag_answer(query, filtered)

    print("\n🟢 Assistant Answer:")
    print(answer)

if __name__ == "__main__":
    chat()
