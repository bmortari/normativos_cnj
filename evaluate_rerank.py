# evaluate_with_reranker.py

import pandas as pd
import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# --- CONFIGURAÇÕES ---
load_dotenv()
K_RETRIEVAL = 10 # Quantos documentos recuperar inicialmente do banco de dados
OUTPUT_DIR = "evaluation_charts_reranked" # Nova pasta para os gráficos com re-ranking

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Modelo de Re-ranking (Cross-Encoder)
RERANKER_MODEL = 'unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2'

# Mapeamento dos experimentos (id e descrição atualizados)
EVALUATION_CONFIGS = [
    {
        "id": "minilm_fixed_reranked",
        "silver_set_table": "public.silver_set_minilm_fixed",
        "vector_table": "public.normativos_cnj_minilm_fixed",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Retriever: MiniLM Fixo + Re-ranker"
    },
    {
        "id": "minilm_structured_reranked",
        "silver_set_table": "public.silver_set_minilm_structured",
        "vector_table": "public.normativos_cnj_minilm_structured",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Retriever: MiniLM Estruturado + Re-ranker"
    },
    {
        "id": "labse_fixed_reranked",
        "silver_set_table": "public.silver_set_labse_fixed",
        "vector_table": "public.normativos_cnj_labse_fixed",
        "model_name": "sentence-transformers/LaBSE",
        "description": "Retriever: LaBSE Fixo + Re-ranker"
    },
    {
        "id": "labse_structured_reranked",
        "silver_set_table": "public.silver_set_labse_structured",
        "vector_table": "public.normativos_cnj_labse_structured",
        "model_name": "sentence-transformers/LaBSE",
        "description": "Retriever: LaBSE Estruturado + Re-ranker"
    }
]

# --- FUNÇÕES DE BANCO E BUSCA (MODIFICADA) ---
def get_db_connection():
    db_params = {
        "host": os.getenv("DB_HOST"), "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME_POSTGRES"), "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASS")
    }
    if not all(db_params.values()):
        raise ValueError("Uma ou mais variáveis de ambiente do banco de dados não foram definidas.")
    return psycopg2.connect(**db_params)

def load_silver_set(conn, silver_table_name):
    query_object = sql.SQL("SELECT id_original, pergunta FROM {}").format(
        sql.Identifier(*silver_table_name.split('.'))
    )
    query_string = query_object.as_string(conn)
    df = pd.read_sql_query(query_string, conn)
    df['id_original'] = df['id_original'].astype(int)
    return df.to_dict('records')

def search_vector_db_with_content(conn, vector_table, query_vector, k) -> List[Dict]:
    """
    Busca no DB e retorna uma lista de dicionários com 'id' e 'document'.
    """
    cursor = conn.cursor()
    # MODIFICAÇÃO: seleciona 'id' e 'document'
    query = sql.SQL("""
        SELECT id, document FROM {table} ORDER BY embedding <=> %s LIMIT %s;
    """).format(table=sql.Identifier(*vector_table.split('.')))
    vector_str = np.array2string(query_vector, separator=',')
    cursor.execute(query, (vector_str, k))
    
    # Retorna uma lista de dicionários
    results = [{'id': item[0], 'document': item[1]} for item in cursor.fetchall()]
    cursor.close()
    return results

# --- FUNÇÕES DE MÉTRICAS E PLOTAGEM (sem alteração) ---
# ... (copie as funções calculate_advanced_metrics, plot_rank_distribution, plot_hit_rate_at_k_curve daqui) ...
def calculate_advanced_metrics(ranks, k):
    successful_queries = len(ranks)
    if successful_queries == 0: return {"Hit@1": 0, f"Hit@{k}": 0, "MRR": 0, f"Precision@{k}": 0, f"Recall@{k}": 0, f"F1-Score@{k}": 0}
    hits_at_1 = sum(1 for r in ranks if r == 1)
    hits_at_k = sum(1 for r in ranks if r <= k)
    mrr_sum = sum(1.0/r for r in ranks if r != float('inf'))
    recall_at_k = hits_at_k / successful_queries
    precision_at_k = (hits_at_k / k) / successful_queries
    f1_score_at_k = 0.0 if (precision_at_k + recall_at_k) == 0 else 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    return {"Hit@1": hits_at_1 / successful_queries, f"Hit@{k}": hits_at_k / successful_queries, "MRR": mrr_sum / successful_queries, f"Precision@{k}": precision_at_k, f"Recall@{k}": recall_at_k, f"F1-Score@{k}": f1_score_at_k}

def plot_rank_distribution(ranks, config, output_dir):
    plt.figure(figsize=(12, 7))
    plot_ranks = [r if r <= K_RETRIEVAL else K_RETRIEVAL + 1 for r in ranks]
    ax = sns.countplot(x=plot_ranks, palette="viridis")
    ax.set_title(f"Distribuição de Ranks para\n{config['description']}", fontsize=16)
    ax.set_xlabel("Rank do Documento Correto", fontsize=12)
    ax.set_ylabel("Contagem de Perguntas", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    tick_labels = [str(i) for i in range(1, K_RETRIEVAL + 1)] + [f'>{K_RETRIEVAL} (Não encontrado)']
    plt.xticks(ticks=range(K_RETRIEVAL + 1), labels=tick_labels)
    filepath = os.path.join(output_dir, f"rank_distribution_{config['id']}.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Gráfico de distribuição de ranks salvo em: {filepath}")

def plot_hit_rate_at_k_curve(all_ranks_data, output_dir):
    plt.figure(figsize=(12, 8))
    for config_id, data in all_ranks_data.items():
        description, ranks = data['description'], data['ranks']
        hit_rates = []
        k_values = range(1, K_RETRIEVAL + 1)
        for k in k_values:
            hits = sum(1 for r in ranks if r <= k)
            hit_rate = (hits / len(ranks)) * 100 if len(ranks) > 0 else 0
            hit_rates.append(hit_rate)
        plt.plot(k_values, hit_rates, marker='o', linestyle='-', label=description)
    plt.title('Performance com Re-ranking: Hit Rate @ K', fontsize=16)
    plt.xlabel('Posição no Ranking (K)', fontsize=12)
    plt.ylabel('Hit Rate (%)', fontsize=12)
    plt.xticks(range(1, K_RETRIEVAL + 1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Configuração', bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    filepath = os.path.join(output_dir, f"hit_rate_at_k_comparison_reranked.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Gráfico comparativo de Hit Rate @ K salvo em: {filepath}")

# --- FUNÇÃO PRINCIPAL ORQUESTRADORA (MODIFICADA) ---

def main():
    summary_results = []
    all_ranks_data = {}
    conn = None

    try:
        conn = get_db_connection()
        
        # Carrega o modelo de re-ranking uma vez
        print(f"Carregando modelo de Re-ranking: {RERANKER_MODEL}...")
        reranker = CrossEncoder(RERANKER_MODEL)
        print("Re-ranker carregado.")

        for config in EVALUATION_CONFIGS:
            print("\n" + "="*80)
            print(f"INICIANDO AVALIAÇÃO PARA: {config['description']}")
            print("="*80)

            retriever_model = SentenceTransformer(config['model_name'])
            test_data = load_silver_set(conn, config['silver_set_table'])
            
            if not test_data:
                print("Nenhum dado de teste. Pulando...")
                continue

            all_ranks = []
            errors = 0
            
            for item in tqdm(test_data, desc=f"Avaliando '{config['id']}'"):
                try:
                    question = item['pergunta']
                    expected_id = item['id_original']

                    # 1. ETAPA DE RETRIEVAL (Bi-Encoder)
                    query_vector = retriever_model.encode(question)
                    retrieved_docs = search_vector_db_with_content(conn, config['vector_table'], query_vector, K_RETRIEVAL)
                    
                    if not retrieved_docs:
                        all_ranks.append(float('inf'))
                        continue
                        
                    # Verifica se o documento esperado está na lista de candidatos
                    # Se não estiver, não há chance do re-ranker encontrá-lo.
                    retrieved_ids = [doc['id'] for doc in retrieved_docs]
                    if expected_id not in retrieved_ids:
                        all_ranks.append(float('inf'))
                        continue
                    
                    # 2. ETAPA DE RE-RANKING (Cross-Encoder)
                    # Prepara os pares (pergunta, documento) para o re-ranker
                    reranker_input = [[question, doc['document']] for doc in retrieved_docs]
                    
                    # Calcula as pontuações de relevância
                    scores = reranker.predict(reranker_input)
                    
                    # Combina os documentos com suas novas pontuações
                    doc_scores = list(zip(retrieved_docs, scores))
                    
                    # Reordena os documentos com base na maior pontuação
                    reranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
                    
                    # Extrai os IDs na nova ordem
                    reranked_ids = [doc['id'] for doc, score in reranked_docs]
                    
                    # 3. AVALIAÇÃO
                    try:
                        rank = reranked_ids.index(expected_id) + 1
                    except ValueError:
                        # Este caso não deveria acontecer por causa da verificação anterior, mas é uma segurança
                        rank = float('inf')
                    all_ranks.append(rank)

                except Exception as e:
                    print(f"\nERRO ao processar id_original {item.get('id_original', 'N/A')}: {e}")
                    errors += 1
            
            # Salva os ranks e gera gráficos (lógica inalterada)
            all_ranks_data[config['id']] = {"description": config['description'], "ranks": all_ranks}
            plot_rank_distribution(all_ranks, config, OUTPUT_DIR)
            metrics = calculate_advanced_metrics(all_ranks, K_RETRIEVAL)
            summary_results.append({
                "Configuração": config['description'],
                "Total Perguntas": len(test_data), "Erros": errors,
                "Hit@1 (%)": f"{metrics['Hit@1']*100:.2f}",
                f"Hit@{K_RETRIEVAL} (%)": f"{metrics[f'Hit@{K_RETRIEVAL}']*100:.2f}",
                "MRR": f"{metrics['MRR']:.4f}",
                f"Precision@{K_RETRIEVAL}": f"{metrics[f'Precision@{K_RETRIEVAL}']:.4f}",
                f"Recall@{K_RETRIEVAL}": f"{metrics[f'Recall@{K_RETRIEVAL}']:.4f}",
                f"F1-Score@{K_RETRIEVAL}": f"{metrics[f'F1-Score@{K_RETRIEVAL}']:.4f}",
            })

    except Exception as e:
        print(f"\nOcorreu um erro crítico: {e}")
    finally:
        if conn: conn.close()

    if all_ranks_data:
        plot_hit_rate_at_k_curve(all_ranks_data, OUTPUT_DIR)

    if summary_results:
        print("\n" + "="*110)
        print("RELATÓRIO FINAL DE PERFORMANCE COM RE-RANKING")
        print("="*110)
        summary_df = pd.DataFrame(summary_results)
        print(summary_df.to_string(index=False))
        print("="*110)

if __name__ == "__main__":
    main()