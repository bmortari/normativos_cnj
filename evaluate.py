# evaluate_rag_advanced.py

import pandas as pd
import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAÇÕES ---
load_dotenv()
K_RESULTS = 10 # Aumentamos para 10 para ter um gráfico de curva mais interessante
OUTPUT_DIR = "evaluation_results" # Pasta para salvar os gráficos e os CSVs

# Cria a pasta de saída se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapeamento dos experimentos (mesmo de antes)
EVALUATION_CONFIGS = [
    {
        "id": "bgem3_fixed",
        "silver_set_table": "public.silver_set_fixed",
        "vector_table": "public.normativos_cnj_bgem3_fixed",
        "model_name": "BAAI/bge-m3",
        "description": "Modelo: bge-m3 | Estratégia: Tamanho Fixo"
    },
    {
        "id": "bgem3_structured",
        "silver_set_table": "public.silver_set_structured",
        "vector_table": "public.normativos_cnj_bgem3_structured",
        "model_name": "BAAI/bge-m3",
        "description": "Modelo: bge-m3 | Estratégia: Estruturada"
    },
    {
        "id": "bgem3_recursive",
        "silver_set_table": "public.silver_set_recursive",
        "vector_table": "public.normativos_cnj_bgem3_recursive",
        "model_name": "BAAI/bge-m3",
        "description": "Modelo: bge-m3 | Estratégia: Recursiva"
    },
    {
        "id": "minilm_fixed",
        "silver_set_table": "public.silver_set_minilm_fixed",
        "vector_table": "public.normativos_cnj_minilm_fixed",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Modelo: MiniLM | Estratégia: Tamanho Fixo"
    },
    {
        "id": "minilm_structured",
        "silver_set_table": "public.silver_set_minilm_structured",
        "vector_table": "public.normativos_cnj_minilm_structured",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Modelo: MiniLM | Estratégia: Estruturada"
    },
    {
        "id": "minilm_recursive",
        "silver_set_table": "public.silver_set_recursive",
        "vector_table": "public.normativos_cnj_minilm_recursive",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Modelo: MiniLM | Estratégia: Recursiva"
    },
    {
        "id": "labse_fixed",
        "silver_set_table": "public.silver_set_fixed",
        "vector_table": "public.normativos_cnj_labse_fixed",
        "model_name": "sentence-transformers/LaBSE",
        "description": "Modelo: LaBSE | Estratégia: Tamanho Fixo"
    },
    {
        "id": "labse_structured",
        "silver_set_table": "public.silver_set_structured",
        "vector_table": "public.normativos_cnj_labse_structured",
        "model_name": "sentence-transformers/LaBSE",
        "description": "Modelo: LaBSE | Estratégia: Estruturada"
    },
    {
        "id": "labse_recursive",
        "silver_set_table": "public.silver_set_recursive",
        "vector_table": "public.normativos_cnj_labse_recursive",
        "model_name": "sentence-transformers/LaBSE",
        "description": "Modelo: LaBSE | Estratégia: Recursiva"
    },
]

# --- FUNÇÕES DE BANCO E BUSCA (sem alteração) ---
def get_db_connection():
    db_params = {
        "host": os.getenv("DB_HOST"), "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME_POSTGRES"), "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASS")
    }
    if not all(db_params.values()):
        raise ValueError("Uma ou mais variáveis de ambiente do banco de dados não foram definidas.")
    
    conn = psycopg2.connect(**db_params)
    # --- CORREÇÃO AQUI ---
    # Habilita o modo autocommit. Cada consulta será executada em sua própria
    # transação, impedindo que um erro contamine as consultas seguintes.
    conn.autocommit = True
    # --- FIM DA CORREÇÃO ---
    return conn

def load_silver_set(conn, silver_table_name):
    """Carrega o conjunto de dados de teste (silver set) do banco de dados."""
    print(f"Carregando dados de teste da tabela '{silver_table_name}'...")
    query_object = sql.SQL("SELECT id_original, pergunta FROM {}").format(
        sql.Identifier(*silver_table_name.split('.'))
    )
    df = pd.read_sql_query(query_object.as_string(conn), conn)
    df['id_original'] = df['id_original'].astype(int)
    print(f"{len(df)} perguntas carregadas.")
    return df.to_dict('records')

def search_vector_db(conn, vector_table, query_vector, k):
    cursor = conn.cursor()
    query = sql.SQL("""
        SELECT id FROM {table} ORDER BY embedding <=> %s LIMIT %s;
    """).format(table=sql.Identifier(*vector_table.split('.')))
    
    # --- CORREÇÃO AQUI ---
    # np.array2string() pode truncar vetores longos com '...'.
    # A conversão para lista e depois para string garante o formato completo '[num, num, ...]'
    # que o pgvector espera.
    vector_str = str(query_vector.tolist())
    # --- FIM DA CORREÇÃO ---
    
    cursor.execute(query, (vector_str, k))
    results = [item[0] for item in cursor.fetchall()]
    cursor.close()
    return results

# --- NOVAS FUNÇÕES DE MÉTRICAS E PLOTAGEM (sem alteração) ---
def calculate_advanced_metrics(ranks, k):
    """Calcula um conjunto completo de métricas de ranking."""
    successful_queries = len(ranks)
    if successful_queries == 0:
        return {
            "Hit@1": 0, f"Hit@{k}": 0, "MRR": 0,
            f"Precision@{k}": 0, f"Recall@{k}": 0, f"F1-Score@{k}": 0
        }

    # Métricas de Ranking
    hits_at_1 = sum(1 for r in ranks if r == 1)
    hits_at_k = sum(1 for r in ranks if r <= k)
    mrr_sum = sum(1.0/r for r in ranks if r != float('inf'))
    
    # Métricas adaptadas de Classificação
    recall_at_k = hits_at_k / successful_queries
    precision_at_k = (hits_at_k / k) / successful_queries if successful_queries > 0 else 0
    
    if (precision_at_k + recall_at_k) == 0:
        f1_score_at_k = 0.0
    else:
        f1_score_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

    return {
        "Hit@1": hits_at_1 / successful_queries,
        f"Hit@{k}": hits_at_k / successful_queries,
        "MRR": mrr_sum / successful_queries,
        f"Precision@{k}": precision_at_k,
        f"Recall@{k}": recall_at_k,
        f"F1-Score@{k}": f1_score_at_k
    }

def plot_rank_distribution(ranks, config, output_dir):
    """Gera e salva um gráfico de barras da distribuição dos ranks."""
    plt.figure(figsize=(12, 7))
    plot_ranks = [r if r <= K_RESULTS else K_RESULTS + 1 for r in ranks]
    ax = sns.countplot(x=plot_ranks, palette="viridis")
    
    ax.set_title(f"Distribuição de Ranks para\n{config['description']}", fontsize=16)
    ax.set_xlabel("Rank do Documento Correto", fontsize=12)
    ax.set_ylabel("Contagem de Perguntas", fontsize=12)
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    tick_labels = [str(i) for i in range(1, K_RESULTS + 1)] + [f'>{K_RESULTS} (Não encontrado)']
    plt.xticks(ticks=range(K_RESULTS + 1), labels=tick_labels)
    
    filepath = os.path.join(output_dir, f"rank_distribution_{config['id']}.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Gráfico de distribuição de ranks salvo em: {filepath}")

def plot_hit_rate_at_k_curve(all_ranks_data, output_dir):
    """Gera e salva um gráfico comparativo de Hit Rate @ K para todas as configs."""
    plt.figure(figsize=(12, 8))
    
    for config_id, data in all_ranks_data.items():
        description = data['description']
        ranks = data['ranks']
        
        hit_rates = []
        k_values = range(1, K_RESULTS + 1)
        
        for k in k_values:
            hits = sum(1 for r in ranks if r <= k)
            hit_rate = (hits / len(ranks)) * 100 if len(ranks) > 0 else 0
            hit_rates.append(hit_rate)
            
        plt.plot(k_values, hit_rates, marker='o', linestyle='-', label=description)

    plt.title('Performance de Recuperação: Hit Rate @ K (Análogo à Curva AUC)', fontsize=16)
    plt.xlabel('Número de Documentos Recuperados (K)', fontsize=12)
    plt.ylabel('Hit Rate (%)', fontsize=12)
    plt.xticks(range(1, K_RESULTS + 1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Configuração', bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    
    filepath = os.path.join(output_dir, f"hit_rate_at_k_comparison.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Gráfico comparativo de Hit Rate @ K salvo em: {filepath}")

# --- FUNÇÃO PRINCIPAL ORQUESTRADORA ---
def main():
    summary_results = []
    all_ranks_data = {}
    
    # <<< ADICIONADO: Lista para armazenar resultados detalhados de cada consulta >>>
    detailed_results_list = []
    
    conn = None

    try:
        conn = get_db_connection()
        for config in EVALUATION_CONFIGS:
            print("\n" + "="*80)
            print(f"INICIANDO AVALIAÇÃO PARA: {config['description']}")
            print("="*80)

            model = SentenceTransformer(config['model_name'])
            test_data = load_silver_set(conn, config['silver_set_table'])
            
            if not test_data:
                print("Nenhum dado de teste. Pulando...")
                continue

            all_ranks = []
            errors = 0
            
            for item in tqdm(test_data, desc=f"Avaliando '{config['id']}'"):
                try:
                    query_vector = model.encode(item['pergunta'])
                    retrieved_ids = search_vector_db(conn, config['vector_table'], query_vector, K_RESULTS)
                    
                    try:
                        rank = retrieved_ids.index(item['id_original']) + 1
                    except ValueError:
                        rank = float('inf')
                    all_ranks.append(rank)

                    # <<< ADICIONADO: Armazena os detalhes desta consulta >>>
                    detailed_results_list.append({
                        'config_id': config['id'],
                        'config_description': config['description'],
                        'question_text': item['pergunta'],
                        'expected_doc_id': item['id_original'],
                        'retrieved_top_k_ids': retrieved_ids,
                        'correct_doc_rank': rank,
                        'is_hit_at_k': rank <= K_RESULTS
                    })

                except Exception as e:
                    print(f"\nERRO ao processar id_original {item.get('id_original', 'N/A')}: {e}")
                    errors += 1
            
            all_ranks_data[config['id']] = { "description": config['description'], "ranks": all_ranks }
            plot_rank_distribution(all_ranks, config, OUTPUT_DIR)

            metrics = calculate_advanced_metrics(all_ranks, K_RESULTS)
            summary_results.append({
                "Configuração": config['description'], "Total Perguntas": len(test_data),
                "Erros": errors,
                # <<< ALTERADO: Salva o valor numérico para facilitar a ordenação no CSV >>>
                "Hit@1": metrics['Hit@1'],
                f"Hit@{K_RESULTS}": metrics[f'Hit@{K_RESULTS}'],
                "MRR": metrics['MRR'],
                f"Precision@{K_RESULTS}": metrics[f'Precision@{K_RESULTS}'],
                f"Recall@{K_RESULTS}": metrics[f'Recall@{K_RESULTS}'],
                f"F1-Score@{K_RESULTS}": metrics[f'F1-Score@{K_RESULTS}'],
            })

    except Exception as e:
        print(f"\nOcorreu um erro crítico: {e}")
    finally:
        if conn:
            conn.close()

    if not summary_results:
        print("Nenhum resultado foi gerado. Encerrando.")
        return

    # --- EXIBIÇÃO E SALVAMENTO DOS RESULTADOS ---
    
    # 1. Gera o gráfico comparativo final
    if all_ranks_data:
        plot_hit_rate_at_k_curve(all_ranks_data, OUTPUT_DIR)

    # 2. Processa e salva os CSVs
    summary_df = pd.DataFrame(summary_results)
    
    # <<< ADICIONADO: Salva o sumário em CSV >>>
    summary_csv_path = os.path.join(OUTPUT_DIR, "evaluation_summary.csv")
    # Ordena o dataframe pela métrica mais importante (Hit@K) antes de salvar
    summary_df_sorted = summary_df.sort_values(by=f"Hit@{K_RESULTS}", ascending=False)
    summary_df_sorted.to_csv(summary_csv_path, index=False, float_format='%.4f')
    print(f"\n[+] Sumário dos resultados salvo em: {summary_csv_path}")
    
    # <<< ADICIONADO: Salva os resultados detalhados em CSV >>>
    if detailed_results_list:
        detailed_df = pd.DataFrame(detailed_results_list)
        detailed_csv_path = os.path.join(OUTPUT_DIR, "evaluation_detailed_results.csv")
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"[+] Resultados detalhados por pergunta salvos em: {detailed_csv_path}")

    # <<< ADICIONADO: Salva os dados brutos de ranks para replicação de gráficos >>>
    if all_ranks_data:
        # Transforma o dicionário em um dataframe 'long format' ideal para análise
        ranks_for_csv = []
        for config_id, data in all_ranks_data.items():
            for i, rank in enumerate(data['ranks']):
                ranks_for_csv.append({
                    'config_id': config_id,
                    'config_description': data['description'],
                    'query_index': i,
                    'rank': rank
                })
        ranks_df = pd.DataFrame(ranks_for_csv)
        ranks_csv_path = os.path.join(OUTPUT_DIR, "evaluation_all_ranks.csv")
        ranks_df.to_csv(ranks_csv_path, index=False)
        print(f"[+] Dados brutos de ranks salvos em: {ranks_csv_path}")

    # 3. Exibe o relatório final no console
    print("\n" + "="*90)
    print("RELATÓRIO FINAL COMPARATIVO DE PERFORMANCE")
    print("="*90)
    # Formata as colunas de métricas para exibição em porcentagem
    for col in ["Hit@1", f"Hit@{K_RESULTS}"]:
        summary_df_sorted[col] = summary_df_sorted[col].apply(lambda x: f"{x*100:.2f}%")
    print(summary_df_sorted.to_string(index=False))
    print("="*90)


if __name__ == "__main__":
    main()