"""
Módulo para web scraping de atos normativos do CNJ.
"""

import random
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config import ROOT_URL, LISTING_URL, PAGE_LIMIT, LINK_LIMIT, DELAY_ENABLED, DELAY_RANGE, MAX_WORKERS
from .text_processing import parse_and_clean_html_content, clean_legal_text, slugify_column_name


def process_link(url: str, session: requests.Session, chunking_function: Callable) -> List[Dict]:
    """
    Baixa, parseia e chunkeia o conteúdo de uma única URL.
    Retorna uma lista de chunks ou uma lista vazia em caso de erro.
    """
    try:
        if DELAY_ENABLED:
            time.sleep(random.uniform(*DELAY_RANGE))
        
        response = session.get(url, timeout=30)
        response.raise_for_status()

        detailed_soup = BeautifulSoup(response.text, 'html.parser')
        main_div = detailed_soup.find('body').find('div')
        if not main_div: 
            return []
            
        data_from_page = {}
        for id_div in main_div.find_all('div', class_='identificacao'):
            column_name = id_div.get_text(strip=True)
            content_div = id_div.find_next_sibling('div')
            if content_div:
                safe_col_name = slugify_column_name(column_name)
                cleaned_content = parse_and_clean_html_content(content_div)
                data_from_page[safe_col_name] = cleaned_content

        if not data_from_page: 
            return []
            
        full_text = "\n".join(filter(None, data_from_page.values()))
        full_text = clean_legal_text(full_text)
        
        chunks = chunking_function(raw_text=full_text, document_id=url)

        # Adiciona metadados completos a cada chunk
        for chunk in chunks:
            chunk['metadata'] = data_from_page.copy()
        
        return chunks

    except requests.exceptions.RequestException as e:
        print(f"Erro ao processar URL {url}: {e}")
        return []


def run_scraper(model, conn, table_name: str, chunking_function: Callable):
    """Orquestra o processo de scraping e inserção de forma concorrente e em lote."""
    from .embeddings import generate_embeddings_for_chunks
    from .database import batch_insert_data
    
    page_number = 1
    total_links_processed = 0
    session = requests.Session()

    while True:
        if PAGE_LIMIT > 0 and page_number > PAGE_LIMIT:
            print(f"\nLimite de {PAGE_LIMIT} página(s) atingido. Encerrando scraping.")
            break
        if LINK_LIMIT > 0 and total_links_processed >= LINK_LIMIT:
            print(f"\nLimite de {LINK_LIMIT} link(s) atingido. Encerrando.")
            break

        current_page_url = LISTING_URL.format(page_number)
        print(f"\n--- Buscando links na página: {current_page_url} ---")
        
        list_page_content = session.get(current_page_url).text
        soup = BeautifulSoup(list_page_content, 'html.parser')
        table_body = soup.find('table', class_='table').find('tbody') if soup.find('table') else None
        
        if not table_body or not table_body.find_all('tr'):
            print("Nenhuma linha encontrada. Fim da paginação.")
            break

        links_to_process = []
        for row in table_body.find_all('tr'):
            if LINK_LIMIT > 0 and (total_links_processed + len(links_to_process)) >= LINK_LIMIT:
                break
            link_tag = row.find('a', href=True)
            if link_tag:
                links_to_process.append(urljoin(ROOT_URL, link_tag['href']))

        if not links_to_process:
            print("Nenhum link novo encontrado nesta página.")
            page_number += 1
            continue

        all_chunks_from_page = []
        # Processamento concorrente dos links
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {
                executor.submit(process_link, url, session, chunking_function): url 
                for url in links_to_process
            }
            
            progress = tqdm(as_completed(future_to_url), total=len(links_to_process), desc=f"Processando Links (Pág {page_number})")
            for future in progress:
                chunks = future.result()
                if chunks:
                    all_chunks_from_page.extend(chunks)
        
        total_links_processed += len(links_to_process)
        
        if not all_chunks_from_page:
            print("Nenhum chunk válido gerado a partir dos links desta página.")
            page_number += 1
            continue

        print(f"  |-> {len(all_chunks_from_page)} chunks gerados de {len(links_to_process)} links.")
        
        # Geração de embeddings em lote
        print(f"  |-> Gerando embeddings para {len(all_chunks_from_page)} chunks de uma vez...")
        data_for_db = generate_embeddings_for_chunks(model, all_chunks_from_page)
        
        # Inserção no banco de dados em lotes
        print(f"  |-> Inserindo {len(data_for_db)} registros no banco de dados em lotes...")
        try:
            batch_insert_data(conn, table_name, data_for_db)
            conn.commit()
            print(f"  |-> Commit realizado. {len(data_for_db)} registros salvos na tabela '{table_name}'.")
        except Exception as e:
            print(f"  Erro na inserção em lote: {e}")
            conn.rollback()

        page_number += 1
