"""
RAG pipeline for natural language DDI explanations.

Combines ChromaDB retrieval (DrugBank + PubChem + DailyMed knowledge base)
with the Claude API to generate pharmacologically-grounded explanations of
predicted drug-drug interactions.

Components:
    - init_rag(): loads enriched chunks, ChromaDB, and Anthropic client at startup.
    - retrieve_context(): hybrid lookup (direct + semantic) for a drug pair.
    - generate_explanation(): builds prompt and calls Claude API.
"""

import json
import logging
import os

import anthropic
import chromadb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state — populated by init_rag() at application startup
# ---------------------------------------------------------------------------
RAG_DIR = "data/rag"
ENRICHED_CHUNKS: dict = {}   # str(ogb_idx) -> enriched text chunk
IDX_TO_NAME: dict = {}       # str(ogb_idx) -> drug name
chroma_collection = None     # ChromaDB collection with 4,266 embedded chunks
client = None                # Anthropic API client


def init_rag() -> None:
    """Initialize all RAG components at application startup.

    Loads in order:
        1. Enriched drug chunks (DrugBank + PubChem + DailyMed, one per drug).
        2. OGB index → drug name mapping for human-readable output.
        3. ChromaDB persistent collection for semantic search.
        4. Anthropic client (requires ANTHROPIC_API_KEY environment variable).

    All components degrade gracefully — missing files log a warning rather
    than raising, so the API starts even if RAG data is partially unavailable.
    """
    global ENRICHED_CHUNKS, IDX_TO_NAME, chroma_collection, client

    # 1. Enriched chunks — one text block per drug combining 3 knowledge sources
    chunks_path = os.path.join(RAG_DIR, "enriched_chunks.json")
    if os.path.exists(chunks_path):
        try:
            with open(chunks_path, "r") as f:
                ENRICHED_CHUNKS = json.load(f)
            logger.info(f"Loaded {len(ENRICHED_CHUNKS)} enriched chunks")
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load enriched chunks: {e}")
    else:
        logger.warning(f"Enriched chunks not found at {chunks_path}")

    # 2. OGB index → drug name mapping
    idx_path = os.path.join(RAG_DIR, "idx_to_name.json")
    if os.path.exists(idx_path):
        try:
            with open(idx_path, "r") as f:
                IDX_TO_NAME = json.load(f)
            logger.info(f"Loaded {len(IDX_TO_NAME)} drug name mappings")
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load drug name mapping: {e}")
    else:
        logger.warning(f"Drug name mapping not found at {idx_path}")

    # 3. ChromaDB — persistent vector store with all-MiniLM-L6-v2 embeddings
    chroma_path = os.path.join(RAG_DIR, "chromadb")
    if os.path.exists(chroma_path):
        try:
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            collections = chroma_client.list_collections()
            if collections:
                chroma_collection = chroma_client.get_collection(collections[0].name)
                logger.info(f"ChromaDB loaded: {chroma_collection.count()} documents")
            else:
                logger.warning("ChromaDB path exists but contains no collections")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
    else:
        logger.warning(f"ChromaDB not found at {chroma_path}")

    # 4. Anthropic client — key must be set as a HuggingFace Space secret
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic client initialized")
    else:
        logger.warning(
            "ANTHROPIC_API_KEY not set — RAG explanations will not be generated. "
            "Set it as a Space secret in HuggingFace."
        )


def get_drug_name(idx: int) -> str:
    """Resolve an OGB drug index to a human-readable drug name.

    Args:
        idx (int): OGB-DDI node index (0–4266).

    Returns:
        str: Drug name from IDX_TO_NAME, or 'Drug_{idx}' if not found.
    """
    return IDX_TO_NAME.get(str(idx), f"Drug_{idx}")


def retrieve_context(drug_a_idx: int, drug_b_idx: int) -> tuple[str, str, str]:
    """Retrieve pharmacological context for a drug pair.

    Uses hybrid retrieval:
        1. Direct lookup: exact match on OGB index in enriched chunks.
           Always succeeds if the drug is in the knowledge base.
        2. Semantic search: ChromaDB query for related drugs with similar
           mechanisms (e.g., shared CYP enzymes, similar targets).

    Args:
        drug_a_idx (int): OGB index of the first drug.
        drug_b_idx (int): OGB index of the second drug.

    Returns:
        Tuple of (context_text, drug_a_name, drug_b_name) where context_text
        is a newline-joined string of all retrieved pharmacological text.
    """
    context_parts = []
    drug_a_name = get_drug_name(drug_a_idx)
    drug_b_name = get_drug_name(drug_b_idx)

    # Direct lookup from enriched chunks (DrugBank + PubChem + DailyMed)
    for idx, name in [(str(drug_a_idx), drug_a_name), (str(drug_b_idx), drug_b_name)]:
        if idx in ENRICHED_CHUNKS:
            context_parts.append(f"=== {name} (idx {idx}) ===\n{ENRICHED_CHUNKS[idx]}")
        else:
            logger.warning(f"No enriched chunk found for {name} (idx={idx})")

    # Semantic search for mechanistically related drugs
    if chroma_collection is not None:
        query = f"interaction between {drug_a_name} and {drug_b_name}"
        try:
            results = chroma_collection.query(query_texts=[query], n_results=3)
            if results and results["documents"]:
                for doc in results["documents"][0]:
                    # Avoid duplicating the direct lookup results
                    if doc not in context_parts:
                        context_parts.append(doc)
        except Exception as e:
            logger.warning(f"ChromaDB semantic search failed: {e}")

    return "\n\n".join(context_parts), drug_a_name, drug_b_name


def generate_explanation(
    drug_a_idx: int,
    drug_b_idx: int,
    prediction: float,
    exai_results: dict | None = None,
) -> dict:
    """Generate a natural language DDI explanation using the Claude API.

    Builds a structured prompt combining the interaction probability,
    pharmacological context from the RAG retrieval, and optional ExAI
    results (perturbation, attention, IG) from the GNN models. Sends
    the prompt to Claude and returns the structured response.

    Args:
        drug_a_idx (int): OGB index of the first drug.
        drug_b_idx (int): OGB index of the second drug.
        prediction (float): Interaction probability from the GNN model [0, 1].
        exai_results (dict | None): Optional explainability analysis output
            to include in the prompt for mechanistic grounding.

    Returns:
        dict with keys:
            drug_a: {'idx': int, 'name': str}
            drug_b: {'idx': int, 'name': str}
            prediction: float
            explanation: str — Claude's natural language explanation
            context_sources: int — number of knowledge sources retrieved
        On failure, returns {'error': str}.
    """
    if not client:
        return {
            "error": (
                "ANTHROPIC_API_KEY not configured. "
                "Set it as a HuggingFace Space secret to enable RAG explanations."
            )
        }

    context, drug_a_name, drug_b_name = retrieve_context(drug_a_idx, drug_b_idx)

    # Truncate context and ExAI results to stay within Claude's context window
    exai_section = (
        f"## Model Explainability Results\n"
        f"{json.dumps(exai_results, indent=2)[:2000]}"
        if exai_results else ""
    )

    prompt = f"""You are a pharmacology expert explaining drug-drug interactions to clinicians.

Based on the following information, explain the predicted interaction between {drug_a_name} and {drug_b_name}.

## Prediction
Interaction probability: {prediction:.4f}

## Pharmacological Context
{context[:4000]}

{exai_section}

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
            "context_sources": len(context.split("===")) - 1,
        }
    except anthropic.APIError as e:
        logger.error(f"Claude API error: {e}")
        return {"error": f"Claude API error: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error in generate_explanation: {e}")
        return {"error": str(e)}

