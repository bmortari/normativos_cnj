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
OUTPUT_DIR = "evaluation_charts" # Pasta para salvar os gráficos

# Cria a pasta de saída se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapeamento dos experimentos (mesmo de antes)
EVALUATION_CONFIGS = [
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
        "id": "labse_fixed",
        "silver_set_table": "public.silver_set_labse_fixed",
        "vector_table": "public.normativos_cnj_labse_fixed",
        "model_name": "sentence-transformers/LaBSE",
        "description": "Modelo: LaBSE | Estratégia: Tamanho Fixo"
    },
    {
        "id": "labse_structured",
        "silver_set_table": "public.silver_set_labse_structured",
        "vector_table": "public.normativos_cnj_labse_structured",
        "model_name": "sentence-transformers/LaBSE",
        "description": "Modelo: LaBSE | Estratégia: Estruturada"
    }
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
    return psycopg2.connect(**db_params)

def load_silver_set(conn, silver_table_name):
    """Carrega o conjunto de dados de teste (silver set) do banco de dados."""
    print(f"Carregando dados de teste da tabela '{silver_table_name}'...")
    query_object = sql.SQL("SELECT id_original, pergunta FROM {}").format(
        sql.Identifier(*silver_table_name.split('.'))
    )
    
    # --- CORREÇÃO AQUI ---
    # Converte o objeto de query para uma string segura usando o contexto da conexão
    query_string = query_object.as_string(conn)
    
    # Passa a string renderizada para o pandas
    df = pd.read_sql_query(query_string, conn)
    # --- FIM DA CORREÇÃO ---
    
    df['id_original'] = df['id_original'].astype(int)
    print(f"{len(df)} perguntas carregadas.")
    return df.to_dict('records')

def search_vector_db(conn, vector_table, query_vector, k):
    cursor = conn.cursor()
    query = sql.SQL("""
        SELECT id FROM {table} ORDER BY embedding <=> %s LIMIT %s;
    """).format(table=sql.Identifier(*vector_table.split('.')))
    vector_str = np.array2string(query_vector, separator=',')
    cursor.execute(query, (vector_str, k))
    results = [item[0] for item in cursor.fetchall()]
    cursor.close()
    return results

# --- NOVAS FUNÇÕES DE MÉTRICAS E PLOTAGEM ---

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
    # Recall@k é o mesmo que Hit@k
    recall_at_k = hits_at_k / successful_queries
    # Precision@k: (Número de acertos / k) / total de buscas
    precision_at_k = (hits_at_k / k) / successful_queries
    
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
    # Substitui 'inf' por um valor maior que K para visualização
    plot_ranks = [r if r <= K_RESULTS else K_RESULTS + 1 for r in ranks]
    ax = sns.countplot(x=plot_ranks, palette="viridis")
    
    ax.set_title(f"Distribuição de Ranks para\n{config['description']}", fontsize=16)
    ax.set_xlabel("Rank do Documento Correto", fontsize=12)
    ax.set_ylabel("Contagem de Perguntas", fontsize=12)
    
    # Adiciona rótulos às barras
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    # Ajusta os ticks do eixo x
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
    plt.tight_layout(rect=[0, 0, 0.8, 1]) # Ajusta o layout para a legenda caber
    
    filepath = os.path.join(output_dir, f"hit_rate_at_k_comparison.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Gráfico comparativo de Hit Rate @ K salvo em: {filepath}")

# --- FUNÇÃO PRINCIPAL ORQUESTRADORA ---

def main():
    summary_results = []
    all_ranks_data = {}
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
                except Exception as e:
                    print(f"\nERRO ao processar id_original {item.get('id_original', 'N/A')}: {e}")
                    errors += 1
            
            # Salva os ranks para o gráfico comparativo
            all_ranks_data[config['id']] = {
                "description": config['description'],
                "ranks": all_ranks
            }
            
            # Gera o gráfico de distribuição de ranks para esta config
            plot_rank_distribution(all_ranks, config, OUTPUT_DIR)

            # Calcula e armazena as métricas avançadas
            metrics = calculate_advanced_metrics(all_ranks, K_RESULTS)
            summary_results.append({
                "Configuração": config['description'],
                "Total Perguntas": len(test_data),
                "Erros": errors,
                "Hit@1 (%)": f"{metrics['Hit@1']*100:.2f}",
                f"Hit@{K_RESULTS} (%)": f"{metrics[f'Hit@{K_RESULTS}']*100:.2f}",
                "MRR": f"{metrics['MRR']:.4f}",
                f"Precision@{K_RESULTS}": f"{metrics[f'Precision@{K_RESULTS}']:.4f}",
                f"Recall@{K_RESULTS}": f"{metrics[f'Recall@{K_RESULTS}']:.4f}",
                f"F1-Score@{K_RESULTS}": f"{metrics[f'F1-Score@{K_RESULTS}']:.4f}",
            })

    except Exception as e:
        print(f"\nOcorreu um erro crítico: {e}")
    finally:
        if conn:
            conn.close()

    # Gera o gráfico comparativo final
    if all_ranks_data:
        plot_hit_rate_at_k_curve(all_ranks_data, OUTPUT_DIR)

    # Exibe o relatório final
    if summary_results:
        print("\n" + "="*90)
        print("RELATÓRIO FINAL COMPARATIVO DE PERFORMANCE")
        print("="*90)
        summary_df = pd.DataFrame(summary_results)
        print(summary_df.to_string(index=False))
        print("="*90)

if __name__ == "__main__":
    main()