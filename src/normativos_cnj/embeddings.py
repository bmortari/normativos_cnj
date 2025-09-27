"""
Módulo para geração de embeddings usando modelos de sentence transformers.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import json


def generate_embeddings_for_chunks(model: SentenceTransformer, chunks: List[Dict]) -> List[Tuple]:
    """
    Gera embeddings para uma lista de chunks e prepara os dados para inserção no banco.
    
    Args:
        model: Modelo SentenceTransformer carregado
        chunks: Lista de chunks com metadados
        
    Returns:
        Lista de tuplas (texto, metadados_json, embedding) prontas para inserção no banco
    """
    if not chunks:
        return []
    
    # Gera embeddings em lote para todos os chunks
    texts_to_embed = [chunk['chunk_text'] for chunk in chunks]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    
    # Prepara os dados para inserção no banco
    data_for_db = []
    for i, chunk in enumerate(chunks):
        metadata_to_insert = chunk['metadata'].copy()
        metadata_to_insert.pop("texto", None)
        metadata_to_insert.update({
            "document_id": chunk.get("document_id"),
            "autor": chunk.get("autor"),
            "tipo_chunk": chunk.get("tipo"),
            "artigo_pai": chunk.get("artigo_pai")
        })
        metadata_json = json.dumps(metadata_to_insert, ensure_ascii=False)
        
        data_for_db.append(
            (chunk['chunk_text'], metadata_json, embeddings[i].tolist())
        )
    
    return data_for_db
