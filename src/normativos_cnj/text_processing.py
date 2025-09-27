"""
Módulo para processamento e limpeza de textos jurídicos.
"""

import re
import unicodedata
from typing import List, Dict, Callable
from bs4 import BeautifulSoup


def slugify_column_name(name: str) -> str:
    """Limpa e padroniza um nome de coluna para ser seguro para SQL (slugify)."""
    if not name: 
        return "coluna_desconhecida"
    text = unicodedata.normalize('NFD', str(name)).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    if not text: 
        return "coluna_desconhecida"
    return text


def parse_and_clean_html_content(content_soup: BeautifulSoup) -> str:
    """Extrai texto puro de um objeto BeautifulSoup, removendo tags indesejadas."""
    if not content_soup: 
        return ""
    for tag in content_soup(['script', 'style', 'a', 'img']):
        tag.decompose()
    for p in content_soup.find_all('p'):
        p.replace_with(p.get_text() + '\n')
    cleaned_text = content_soup.get_text(separator=' ', strip=True)
    return cleaned_text.replace('\n ', '\n').replace(' \n', '\n')


def clean_legal_text(text: str) -> str:
    """Aplica limpeza final em textos jurídicos para remover ruídos comuns."""
    if not text: 
        return ""
    text = re.sub(r'^\s*[\.]{5,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\.{5,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_structured_legal_text(raw_text: str, document_id: str, **kwargs) -> List[Dict]:
    """Divide o texto legal usando uma abordagem de 'scanner' baseada em marcadores jurídicos."""
    final_chunks = []
    autor = "Não identificado"
    signature_pattern = r'[\n\s]+([A-Z\s]{5,})\s*?\n\s*?(Ministro|Presidente|Corregedor|Relator|Conselheiro)[\s\S]*'
    signature_match = re.search(signature_pattern, raw_text, re.MULTILINE | re.IGNORECASE)
    if signature_match:
        autor_block = signature_match.group(0).strip()
        autor = autor_block.split('\n')[0].strip()
        raw_text = raw_text[:signature_match.start()].strip()
    
    master_pattern = re.compile(r'(CONSIDERANDO|RESOLVE:|DETERMINA:|Art\.\s*\d+º(?:-A|-B)?\.?|Parágrafo\s+único\.?|§\s*\d+º\.?)', re.IGNORECASE)
    matches = list(master_pattern.finditer(raw_text))
    
    if not matches:
        if raw_text.strip():
            final_chunks.append({
                "document_id": document_id, 
                "autor": autor, 
                "tipo": "Corpo_Unico", 
                "artigo_pai": None, 
                "chunk_text": raw_text.strip()
            })
        return final_chunks
    
    first_match_start = matches[0].start()
    if first_match_start > 0:
        preambulo_text = raw_text[:first_match_start].strip()
        if preambulo_text:
            final_chunks.append({
                "document_id": document_id, 
                "autor": autor, 
                "tipo": "Preambulo", 
                "artigo_pai": None, 
                "chunk_text": preambulo_text
            })
    
    artigo_pai_atual = None
    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        chunk_text = raw_text[start_pos:end_pos].strip()
        marker_text = match.group(1).strip()
        tipo_chunk = "Desconhecido"
        
        if 'art' in marker_text.lower():
            tipo_chunk = "Artigo"
            artigo_pai_atual = re.match(r'^(Art\.\s*\d+º(?:-A|-B)?)', marker_text, re.IGNORECASE).group(1)
        elif 'considerando' in marker_text.lower(): 
            tipo_chunk = "Considerando"
        elif 'parágrafo' in marker_text.lower() or '§' in marker_text: 
            tipo_chunk = "Paragrafo"
        elif 'resolve' in marker_text.lower() or 'determina' in marker_text.lower():
            tipo_chunk = "Resolucao"
            if i + 1 < len(matches): 
                continue
        
        if chunk_text:
            final_chunks.append({
                "document_id": document_id, 
                "autor": autor, 
                "tipo": tipo_chunk, 
                "artigo_pai": artigo_pai_atual, 
                "chunk_text": chunk_text
            })
    
    MIN_CHUNK_SIZE = 30
    return [chunk for chunk in final_chunks if len(chunk['chunk_text']) > MIN_CHUNK_SIZE]


def chunk_fixed_size(raw_text: str, document_id: str, chunk_size=1024, chunk_overlap=100, **kwargs) -> List[Dict]:
    """Divide o texto em pedaços de tamanho fixo com sobreposição."""
    final_chunks = []
    start = 0
    text_len = len(raw_text)
    chunk_id = 0
    
    while start < text_len:
        end = start + chunk_size
        chunk_text = raw_text[start:end]
        if chunk_text.strip():
            final_chunks.append({
                "document_id": document_id, 
                "autor": "Não identificado", 
                "tipo": "Fixo", 
                "artigo_pai": f"chunk_{chunk_id}", 
                "chunk_text": chunk_text
            })
        start += chunk_size - chunk_overlap
        chunk_id += 1
    
    return final_chunks


# Estratégias de chunking disponíveis
CHUNKING_STRATEGIES = {
    "structured": chunk_structured_legal_text, 
    "fixed": chunk_fixed_size
}
