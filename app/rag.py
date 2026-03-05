"""
RAG endpoint for natural language DDI explanations.
Combines model predictions + ChromaDB retrieval + Claude API.
"""
import os
import json
import logging
import anthropic
import chromadb

logger = logging.getLogger(__name__)

# Load data at module level
RAG_DIR = "data/rag"
ENRICHED_CHUNKS = {}
IDX_TO_NAME = {}
chroma_collection = None
client = None


def init_rag():
    """Initialize RAG components."""
    global ENRICHED_CHUNKS, IDX_TO_NAME, chroma_collection, client

    # Load enriched chunks
    chunks_path = os.path.join(RAG_DIR, "enriched_chunks.json")
    if os.path.exists(chunks_path):
        with open(chunks_path, "r") as f:
            ENRICHED_CHUNKS = json.load(f)
        logger.info(f"Loaded {len(ENRICHED_CHUNKS)} enriched chunks")

    # Load idx to name mapping
    idx_path = os.path.join(RAG_DIR, "idx_to_name.json")
    if os.path.exists(idx_path):
        with open(idx_path, "r") as f:
            IDX_TO_NAME = json.load(f)
        logger.info(f"Loaded {len(IDX_TO_NAME)} drug name mappings")

    # Load ChromaDB
    chroma_path = os.path.join(RAG_DIR, "chromadb")
    if os.path.exists(chroma_path):
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        collections = chroma_client.list_collections()
        if collections:
            chroma_collection = chroma_client.get_collection(collections[0].name)
            logger.info(f"ChromaDB loaded: {chroma_collection.count()} documents")

    # Init Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic client initialized")
    else:
        logger.warning("ANTHROPIC_API_KEY not set — RAG will not generate explanations")


def get_drug_name(idx):
    return IDX_TO_NAME.get(str(idx), f"Drug_{idx}")


def retrieve_context(drug_a_idx, drug_b_idx):
    """Retrieve pharmacological context for a drug pair."""
    context_parts = []

    drug_a_name = get_drug_name(drug_a_idx)
    drug_b_name = get_drug_name(drug_b_idx)

    # Direct lookup from enriched chunks
    for idx, name in [(str(drug_a_idx), drug_a_name), (str(drug_b_idx), drug_b_name)]:
        if idx in ENRICHED_CHUNKS:
            context_parts.append(f"=== {name} (idx {idx}) ===\n{ENRICHED_CHUNKS[idx]}")

    # Semantic search for related context
    if chroma_collection:
        query = f"interaction between {drug_a_name} and {drug_b_name}"
        try:
            results = chroma_collection.query(query_texts=[query], n_results=3)
            if results and results["documents"]:
                for doc in results["documents"][0]:
                    if doc not in context_parts:
                        context_parts.append(doc)
        except Exception as e:
            logger.warning(f"ChromaDB search failed: {e}")

    return "\n\n".join(context_parts), drug_a_name, drug_b_name


def generate_explanation(drug_a_idx, drug_b_idx, prediction, exai_results=None):
    """Generate natural language explanation using Claude."""
    if not client:
        return {"error": "ANTHROPIC_API_KEY not configured. Set it as a Space secret."}

    context, drug_a_name, drug_b_name = retrieve_context(drug_a_idx, drug_b_idx)

    prompt = f"""You are a pharmacology expert explaining drug-drug interactions to clinicians.

Based on the following information, explain the predicted interaction between {drug_a_name} and {drug_b_name}.

## Prediction
Interaction probability: {prediction:.4f}

## Pharmacological Context
{context[:4000]}

{f"## Model Explainability Results{chr(10)}{json.dumps(exai_results, indent=2)[:2000]}" if exai_results else ""}

Provide a clear, concise explanation covering:
1. Whether these drugs are likely to interact and why
2. The biological mechanism (enzyme-mediated, target-mediated, etc.)
3. Clinical significance
Keep it under 200 words."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "drug_a": {"idx": drug_a_idx, "name": drug_a_name},
            "drug_b": {"idx": drug_b_idx, "name": drug_b_name},
            "prediction": prediction,
            "explanation": response.content[0].text,
            "context_sources": len(context.split("===")) - 1
        }
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return {"error": str(e)}
