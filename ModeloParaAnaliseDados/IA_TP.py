{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "q2LKOmqg6nOx"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.cluster import KMeans\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.metrics import mean_squared_error\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.metrics import r2_score\n",
        "from sklearn.model_selection import cross_val_score\n",
        "import math"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Pré-processamento**\n",
        "\n"
      ],
      "metadata": {
        "id": "sDzF6K7JTcEl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# leitura dos arquivos\n",
        "disciplinasPCurso = pd.read_csv('disciplinas-por-curso-2021.2.csv', encoding='ISO-8859-1')\n",
        "evadidos = pd.read_csv('evadidos-2021.2.csv', encoding='ISO-8859-1')"
      ],
      "metadata": {
        "id": "5kxpy7_V70R3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Selecionar características relevantes p/ análise, remover dados ausentes e outras manipulações\n",
        "disciplinasPCurso['Desempenho'] =  (disciplinasPCurso['Aprovados'] / disciplinasPCurso['Matriculados'] ) * 100\n",
        "disciplinasPCurso = disciplinasPCurso.drop(['Ano', 'Semestre', 'Departamento', 'Codigo Disciplina','Disciplina', 'Reprovacoes', 'Trancamentos', 'Unnamed: 10', 'Aprovados', 'Matriculados'], axis=1)\n",
        "print(disciplinasPCurso)"
      ],
      "metadata": {
        "id": "ve4kJbjZ9fiV",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0aad12ff-1050-4081-bb89-c328ba555f5a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                            Unidade Academica  Desempenho\n",
            "0     CENTRO DE EDUCACAO ABERTA E A DISTANCIA   28.888889\n",
            "1     CENTRO DE EDUCACAO ABERTA E A DISTANCIA   66.666667\n",
            "2     CENTRO DE EDUCACAO ABERTA E A DISTANCIA  100.000000\n",
            "3     CENTRO DE EDUCACAO ABERTA E A DISTANCIA   95.000000\n",
            "4     CENTRO DE EDUCACAO ABERTA E A DISTANCIA   70.000000\n",
            "...                                       ...         ...\n",
            "1818                 UNIDADE CURSO SEQUENCIAL   82.608696\n",
            "1819                 UNIDADE CURSO SEQUENCIAL   88.235294\n",
            "1820                 UNIDADE CURSO SEQUENCIAL   56.250000\n",
            "1821                 UNIDADE CURSO SEQUENCIAL   66.666667\n",
            "1822                 UNIDADE CURSO SEQUENCIAL   40.909091\n",
            "\n",
            "[1823 rows x 2 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.DataFrame(disciplinasPCurso)\n",
        "\n",
        "# Exportando para um arquivo CSV\n",
        "df.to_csv('saida_dados.csv', index=False)\n",
        "# Exportando para um arquivo Excel\n",
        "df.to_excel('saida_dados.xlsx', index=False)"
      ],
      "metadata": {
        "id": "8OBlS3xAN-Qh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Agrupando por Unidade Academica e fazendo a média do desempenho de cada unidade\n",
        "media_desempenho = disciplinasPCurso.groupby('Unidade Academica').mean().reset_index()\n",
        "print(media_desempenho)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6grvYkvcSHX2",
        "outputId": "f928a809-f9fd-4247-ddec-55cec5562ab7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                            Unidade Academica  Desempenho\n",
            "0     CENTRO DE EDUCACAO ABERTA E A DISTANCIA   47.653322\n",
            "1      ESCOLA DE DIREITO TURISMO E MUSEOLOGIA   76.440794\n",
            "2                          ESCOLA DE FARMACIA   81.568356\n",
            "3                          ESCOLA DE MEDICINA   84.877641\n",
            "4                             ESCOLA DE MINAS   75.098965\n",
            "5                          ESCOLA DE NUTRICAO   79.815969\n",
            "6    INSTITUTO DE CIENCIAS EXATAS E APLICADAS   65.035382\n",
            "7   INSTITUTO DE CIENCIAS EXATAS E BIOLOGICAS   67.652070\n",
            "8     INSTITUTO DE CIENCIAS HUMANAS E SOCIAIS   73.760915\n",
            "9     INSTITUTO DE CIENCIAS SOCIAIS APLICADAS   71.062935\n",
            "10     INSTITUTO DE FILOSOFIA ARTES E CULTURA   79.797663\n",
            "11                                   REITORIA   80.669493\n",
            "12                   UNIDADE CURSO SEQUENCIAL   68.619607\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Adicionando rótulos a planilha de evadidos porque ela originalmente não possui rotulação\n",
        "rotulos = ['Ano', 'Periodo', 'Cidade', 'Unidade Academica', 'Curso' , 'Modalidade', 'Presenca', 'Cancelamento', 'TotalEvasao' , 'Branco']\n",
        "evadidos.columns = rotulos\n",
        "# Tirando os rótulos que não contém informações pertinentes\n",
        "evadidos = evadidos.drop(['Ano', 'Periodo' ,'Cidade', 'Curso', 'Modalidade', 'Presenca', 'Cancelamento','Branco'], axis=1)\n",
        "# Agrupando por unidade academia, o mesmo feito com a planilha de desempenho\n",
        "media_desempenho = evadidos.groupby('Unidade Academica').mean().reset_index()\n",
        "media_desempenho['TotalEvasao'] = media_desempenho['TotalEvasao'].apply(lambda x: math.ceil(x))\n",
        "print(media_desempenho)\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bCPJ1O35WtE6",
        "outputId": "c54d42d2-4f37-4c91-9acf-b635f1c46dae"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                            Unidade Academica  TotalEvasao\n",
            "0     CENTRO DE EDUCACAO ABERTA E A DISTANCIA            4\n",
            "1      ESCOLA DE DIREITO TURISMO E MUSEOLOGIA            8\n",
            "2                   ESCOLA DE EDUCACAO FISICA            4\n",
            "3                          ESCOLA DE FARMACIA            8\n",
            "4                          ESCOLA DE MEDICINA            8\n",
            "5                             ESCOLA DE MINAS            7\n",
            "6                          ESCOLA DE NUTRICAO            8\n",
            "7    INSTITUTO DE CIENCIAS EXATAS E APLICADAS            9\n",
            "8   INSTITUTO DE CIENCIAS EXATAS E BIOLOGICAS            5\n",
            "9     INSTITUTO DE CIENCIAS HUMANAS E SOCIAIS            5\n",
            "10    INSTITUTO DE CIENCIAS SOCIAIS APLICADAS            6\n",
            "11     INSTITUTO DE FILOSOFIA ARTES E CULTURA            3\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "dfE = pd.DataFrame(evadidos)\n",
        "\n",
        "# Exportando para um arquivo CSV\n",
        "dfE.to_csv('saida_dadosEvadido.csv', index=False)  # O parâmetro index=False evita que a coluna de índices seja salva no arquivo\n",
        "# Exportando para um arquivo Excel\n",
        "dfE.to_excel('saida_dadosEvadido.xlsx', index=False)  # O parâmetro index=False evita que a coluna de índices seja salva no arquivo\n"
      ],
      "metadata": {
        "id": "ivY-WkSkahXo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Clusterização**"
      ],
      "metadata": {
        "id": "ZJe66dtvouPW"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Carregar os dados de evasão e taxa de aprovados\n",
        "evasao_data = pd.read_csv('saida_dadosEvadido.csv')\n",
        "aprovados_data = pd.read_csv('saida_dados.csv')"
      ],
      "metadata": {
        "id": "mM86M_FLklGm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Selecionar apenas as colunas relevantes para a clusterização\n",
        "evasao_data = evasao_data[['Unidade Academica', 'TotalEvasao']]\n",
        "aprovados_data = aprovados_data[['Unidade Academica', 'Desempenho']]"
      ],
      "metadata": {
        "id": "ihVi7VIcknof"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Combinar os dados de evasão e taxa de aprovados em um único DataFrame\n",
        "merged_data = pd.merge(evasao_data, aprovados_data, on='Unidade Academica')"
      ],
      "metadata": {
        "id": "ERb_Mfk4ky_m"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Definir as features (variáveis de entrada) para a clusterização\n",
        "X = merged_data[['TotalEvasao', 'Desempenho']].copy()\n",
        "\n",
        "# Calcular a média de cada coluna\n",
        "media_total_evasao = X['TotalEvasao'].mean()\n",
        "media_desempenho = X['Desempenho'].mean()\n",
        "\n",
        "# Preencher os valores igual a 0 pela média da coluna\n",
        "X['TotalEvasao'].replace(0, media_total_evasao, inplace=True)\n",
        "X['Desempenho'].replace(0, media_desempenho, inplace=True)\n",
        "\n",
        "# Preencher valores ausentes com a média da coluna\n",
        "X.fillna(X.mean(), inplace=True)\n",
        "\n",
        "# Reiniciar o índice do DataFrame após remover as linhas com valores ausentes\n",
        "X.reset_index(drop=True, inplace=True)\n",
        "\n",
        "print(X)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WLcXyp9Ikn-W",
        "outputId": "7de65cf8-6185-43a4-f8bc-74bf951a75f8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "       TotalEvasao  Desempenho\n",
            "0                3   28.888889\n",
            "1                3   66.666667\n",
            "2                3  100.000000\n",
            "3                3   95.000000\n",
            "4                3   70.000000\n",
            "...            ...         ...\n",
            "23815            8   47.619048\n",
            "23816            8   72.078107\n",
            "23817            8   72.078107\n",
            "23818            8   71.428571\n",
            "23819            8   97.435897\n",
            "\n",
            "[23820 rows x 2 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Normalizar os dados para garantir que todas as variáveis tenham a mesma escala\n",
        "scaler = StandardScaler()\n",
        "X_normalized = scaler.fit_transform(X)"
      ],
      "metadata": {
        "id": "7eK0qVv9mNLh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Definir o número de clusters (K) que você deseja obter\n",
        "K = 3  # Número de clusters desejado"
      ],
      "metadata": {
        "id": "jBcVSTanmS6q"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Criar o modelo K-Means\n",
        "kmeans = KMeans(n_clusters=K, random_state=42)"
      ],
      "metadata": {
        "id": "OZYBOeTJmpK6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Realizar a clusterização\n",
        "clusters = kmeans.fit_predict(X_normalized)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cqTHzkytmvsB",
        "outputId": "53ef232a-f506-41c7-b636-7486c283415b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Adicionar os rótulos dos clusters ao DataFrame\n",
        "merged_data['Cluster'] = clusters"
      ],
      "metadata": {
        "id": "baGQSWf_qvC8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Criar uma cópia dos dados para evitar modificar os dados originais\n",
        "X_scaled = X.copy()\n",
        "\n",
        "# Escalando as variáveis para o intervalo [0, 1]\n",
        "scaler = MinMaxScaler()\n",
        "X_scaled[['TotalEvasao', 'Desempenho']] = scaler.fit_transform(X[['TotalEvasao', 'Desempenho']])\n",
        "\n",
        "# Realizar a clusterização novamente nos dados escalados\n",
        "kmeans = KMeans(n_clusters=3, random_state=42)\n",
        "clusters = kmeans.fit_predict(X_scaled)\n",
        "\n",
        "# Visualizar os resultados\n",
        "plt.scatter(X_scaled['TotalEvasao'], X_scaled['Desempenho'], c=clusters, cmap='viridis')\n",
        "plt.xlabel('Total de Evasão (Escalado)')\n",
        "plt.ylabel('Taxa de Aprovados (Escalada)')\n",
        "plt.title('Clusterização dos Cursos')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "yJaVAlAnujS_",
        "outputId": "d6f06a6f-808f-4461-da98-b56b888d00ee",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 526
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHHCAYAAABDUnkqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAADtO0lEQVR4nOydd3gU5fbHvzPb03tCaKH3JkhHVEAUQcUGWPCiqFdFEVQUG4K9YQX5oWL3YkPFRhWsSG/SWwgkJJDes2Xe3x8hkZhk52w4mZ1N5vM8PPeafTPnze7szJlTvkcSQggYGBgYGBgYGDQQZH9vwMDAwMDAwMCAE8O5MTAwMDAwMGhQGM6NgYGBgYGBQYPCcG4MDAwMDAwMGhSGc2NgYGBgYGDQoDCcGwMDAwMDA4MGheHcGBgYGBgYGDQoDOfGwMDAwMDAoEFhODcGBgYGBgYGDQrDuTEw0ICkpCT85z//8fc26sT7778PSZKQnJzs132MHz8eoaGhuP/++5GTk4OIiAjk5ubWu129/P0GBgZ0DOfGwOAsOHToEG6//Xa0bt0adrsdYWFhGDRoEF577TWUlJRosofi4mI88cQTWLt2rSb2/MHu3buxdu1azJ49G0uXLkV0dDSGDx+OiIgIf2+tXlm7di2uvPJKJCQkwGq1Ii4uDmPGjMGSJUv8vTUDA11j9vcGDAwClR9++AHXXHMNbDYbJk6ciK5du8LpdOL333/HAw88gF27dmHhwoX1vo/i4mLMnj0bAHD++eezH//GG2/E+PHjYbPZ2I9NpXXr1ti8eTOaNm2Ke++9F+np6WjSpInf9qMFs2bNwpw5c9CuXTvcfvvtaNmyJbKysvDjjz/iqquuwieffILrrrvO39s0MNAlhnNjYFAHjhw5gvHjx6Nly5b4+eefq9xo77rrLhw8eBA//PCDH3d49hQVFSE4OBgmkwkmk8mve7Hb7WjatCkAQJZlJCYm+nU/9c2XX36JOXPm4Oqrr8ann34Ki8VS+doDDzyA5cuXw+VysdgqLi5GUFAQy7EMDPSCkZYyMKgDL7zwAgoLC/Huu+/WGEFo27Ytpk6dWuvvP/HEE5AkqdrPa6rv2LRpE0aOHImYmBg4HA60atUKN998MwAgOTkZsbGxAIDZs2dDkiRIkoQnnnii8vf37t2Lq6++GlFRUbDb7ejTpw+WLl1ao91ffvkFd955J+Li4tCsWbMa91Sx95r+nVlX9NJLL2HgwIGIjo6Gw+FA79698eWXX9b4fnz88cfo27cvgoKCEBkZifPOOw8rVqyofP3rr7/GqFGjkJiYCJvNhjZt2uDJJ5+Ex+OpdqwvvvgCvXv3hsPhQExMDG644QakpqbW+lmcya5du3DhhRfC4XCgWbNmeOqpp6AoSo1r58+fjy5dusBmsyExMRF33XVXtRqgAwcO4KqrrkJCQgLsdjuaNWuG8ePHIy8vz+s+HnvsMURFRWHRokVVHJsKRo4cidGjRwOovSZo7dq1kCSpSrry/PPPR9euXbF582acd955CAoKwsMPPwzA+3lWQVFREe677z40b94cNpsNHTp0wEsvvQQhRJV1K1euxODBgxEREYGQkBB06NCh0o6BgRYYkRsDgzrw3XffoXXr1hg4cGC92jl58iQuuugixMbG4qGHHkJERASSk5Mray5iY2Px1ltv4Y477sDYsWNx5ZVXAgC6d+8OoPxmPWjQIDRt2hQPPfQQgoOD8fnnn+OKK67AV199hbFjx1axd+eddyI2NhaPP/44ioqKatzTlVdeibZt21b52ebNm/Hqq68iLi6u8mevvfYaLrvsMlx//fVwOp1YvHgxrrnmGnz//fe49NJLK9fNnj0bTzzxBAYOHIg5c+bAarVi/fr1+Pnnn3HRRRcBABYtWoTQ0FBMnz4dwcHBWLNmDR5//HHk5+fjxRdfrDzW+++/j0mTJuHcc8/Fs88+i4yMDLz22mv4448/sHXrVq81Ounp6bjgggvgdrsr36uFCxfC4XBUW/vEE09g9uzZGD58OO644w7s27cPb731FjZu3Ig//vgDFosFTqcTI0eORFlZGe6++24kJCQgNTUV33//PXJzcxEeHl7jPg4cOIC9e/fi5ptvRmhoaK37rStZWVm45JJLMH78eNxwww2Ij49XPc8AQAiByy67DGvWrMEtt9yCnj17Yvny5XjggQeQmpqKV155BUD5OTd69Gh0794dc+bMgc1mw8GDB/HHH3+w/y0GBrUiDAwMfCIvL08AEJdffjn5d1q2bCluuummyv+eNWuWqOnr99577wkA4siRI0IIIb7++msBQGzcuLHWY586dUoAELNmzar22rBhw0S3bt1EaWlp5c8URREDBw4U7dq1q2Z38ODBwu12e91TTfZbtGghunXrJgoL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0Ufx7bffVntfBgwYgEsvvRRz587FxRdfjOuuuw4nT57EvHnz0LZtW+zYsaNyfdu2bfHII4/gySefxJAhQ3DllVfCZrNh48aNSExMxLPPPouBAwciMjISN910E+655x5IkoSPPvqo2s3YYrHg+eefx6RJkzB06FBMmDChshU8KSkJ06ZN8/p3zZgxAx999BEuvvhiTJ06tbIVvGXLllX2HBsbi5kzZ2L27Nm4+OKLcdlll2Hfvn2YP38+zj333MpaoJ9//hlTpkzBNddcg/bt28PtduOjjz6CyWTCVVdd5XUv48aNw86dO/H0009j69atmDBhQqVC8bJly7B69Wp8+umnAIAuXbqgf//+mDlzJrKzsxEVFYXFixfX6BTVBuU8GzNmDC644AI88sgjSE5ORo8ePbBixQp8++23uPfeeyudrDlz5uDXX3/FpZdeipYtW+LkyZOYP38+mjVrhsGDB5P3ZGBwVvitT8vAoAGwf/9+ceutt4qkpCRhtVpFaGioGDRokHjjjTeqtF//uxVcCCE2b94s+vXrJ6xWq2jRooWYO3dutbbeLVu2iAkTJogWLVoIm80m4uLixOjRo8WmTZuqHOvPP/8UvXv3FlartVpb+KFDh8TEiRNFQkKCsFgsomnTpmL06NHiyy+/rFxTYbemVuB/7+mmm24SAGr8d+bf+O6774p27doJm80mOnbsKN57771aW+AXLVokevXqVXmcoUOHipUrV1a+/scff4j+/fsLh8MhEhMTxYwZM8Ty5curtToLIcRnn30mevXqJWw2m4iKihLXX3+9OH78eE0fXzV27Nghhg4dKux2u2jatKl48sknxbvvvltjq/Wbb74pOnbsKCwWi4iPjxd33HGHyMnJqXz98OHD4uabbxZt2rQRdrtdREVFiQsuuECsWrWKtBchhFi9erW4/PLLRVxcnDCbzSI2NlaMGTNGfPvtt1XWHTp0SAwfPlzYbDYRHx8vHn74YbFy5coaW8G7dOlSzQ71PCsoKBDTpk0TiYmJwmKxiHbt2okXX3yxStt+xZ4TExOF1WoViYmJYsKECWL//v3kv9vA4GyRhPCxus/AwMCgnkhOTsaIESOwa9euah09BgYGBlSMmhsDAwPdkJSUhJCQEPz+++/+3oqBgUEAY9TcGBgY6IInnngCMTExOHDggM/txwYGBgZnYqSlDAwMdEHr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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Adicionar os rótulos dos clusters ao DataFrame\n",
        "merged_data['Cluster'] = clusters\n",
        "\n",
        "# Filtrar os cursos por cluster\n",
        "cluster_0_courses = merged_data[merged_data['Cluster'] == 0]\n",
        "cluster_1_courses = merged_data[merged_data['Cluster'] == 1]\n",
        "cluster_2_courses = merged_data[merged_data['Cluster'] == 2]\n",
        "\n",
        "# Salvar os DataFrames em arquivos CSV ou Excel\n",
        "\n",
        "cluster0Aprox = cluster_0_courses.groupby('Unidade Academica').mean().reset_index()\n",
        "cluster0Aprox.to_csv('cursos_cluster_0.csv', index=False)\n",
        "\n",
        "cluster1Aprox = cluster_1_courses.groupby('Unidade Academica').mean().reset_index()\n",
        "cluster1Aprox.to_csv('cursos_cluster_1.csv', index=False)\n",
        "\n",
        "cluster2Aprox = cluster_2_courses.groupby('Unidade Academica').mean().reset_index()\n",
        "cluster2Aprox.to_csv('cursos_cluster_2.csv', index=False)\n",
        "\n",
        "\n",
        "# Exibir os cursos de cada cluster\n",
        "print(\"Cursos do Cluster 0:\")\n",
        "print(cluster_0_courses)\n",
        "\n",
        "print(\"\\nCursos do Cluster 1:\")\n",
        "print(cluster_1_courses)\n",
        "\n",
        "print(\"\\nCursos do Cluster 2:\")\n",
        "print(cluster_2_courses)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_oAOV3TI8VwW",
        "outputId": "b46a305d-d318-4ee2-a669-17c4ef9ac4e9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cursos do Cluster 0:\n",
            "                             Unidade Academica  TotalEvasao  Desempenho  \\\n",
            "0      CENTRO DE EDUCACAO ABERTA E A DISTANCIA            3   28.888889   \n",
            "7      CENTRO DE EDUCACAO ABERTA E A DISTANCIA            3   28.323699   \n",
            "8      CENTRO DE EDUCACAO ABERTA E A DISTANCIA            3   27.745665   \n",
            "9      CENTRO DE EDUCACAO ABERTA E A DISTANCIA            3   25.287356   \n",
            "10     CENTRO DE EDUCACAO ABERTA E A DISTANCIA            3   26.136364   \n",
            "...                                        ...          ...         ...   \n",
            "23788                       ESCOLA DE MEDICINA            8   49.019608   \n",
            "23789                       ESCOLA DE MEDICINA            8   52.941176   \n",
            "23790                       ESCOLA DE MEDICINA            8   46.666667   \n",
            "23797                       ESCOLA DE MEDICINA            8   54.838710   \n",
            "23815                       ESCOLA DE MEDICINA            8   47.619048   \n",
            "\n",
            "       Cluster  \n",
            "0            0  \n",
            "7            0  \n",
            "8            0  \n",
            "9            0  \n",
            "10           0  \n",
            "...        ...  \n",
            "23788        0  \n",
            "23789        0  \n",
            "23790        0  \n",
            "23797        0  \n",
            "23815        0  \n",
            "\n",
            "[4846 rows x 4 columns]\n",
            "\n",
            "Cursos do Cluster 1:\n",
            "                             Unidade Academica  TotalEvasao  Desempenho  \\\n",
            "1      CENTRO DE EDUCACAO ABERTA E A DISTANCIA            3   66.666667   \n",
            "2      CENTRO DE EDUCACAO ABERTA E A DISTANCIA            3  100.000000   \n",
            "3      CENTRO DE EDUCACAO ABERTA E A DISTANCIA            3   95.000000   \n",
            "4      CENTRO DE EDUCACAO ABERTA E A DISTANCIA            3   70.000000   \n",
            "5      CENTRO DE EDUCACAO ABERTA E A DISTANCIA            3   90.000000   \n",
            "...                                        ...          ...         ...   \n",
            "23754  INSTITUTO DE CIENCIAS HUMANAS E SOCIAIS            1  100.000000   \n",
            "23755  INSTITUTO DE CIENCIAS HUMANAS E SOCIAIS            1   80.000000   \n",
            "23756  INSTITUTO DE CIENCIAS HUMANAS E SOCIAIS            1   90.000000   \n",
            "23757  INSTITUTO DE CIENCIAS HUMANAS E SOCIAIS            1   66.666667   \n",
            "23758  INSTITUTO DE CIENCIAS HUMANAS E SOCIAIS            1   69.014085   \n",
            "\n",
            "       Cluster  \n",
            "1            1  \n",
            "2            1  \n",
            "3            1  \n",
            "4            1  \n",
            "5            1  \n",
            "...        ...  \n",
            "23754        1  \n",
            "23755        1  \n",
            "23756        1  \n",
            "23757        1  \n",
            "23758        1  \n",
            "\n",
            "[11745 rows x 4 columns]\n",
            "\n",
            "Cursos do Cluster 2:\n",
            "                             Unidade Academica  TotalEvasao  Desempenho  \\\n",
            "79     CENTRO DE EDUCACAO ABERTA E A DISTANCIA            9   66.666667   \n",
            "80     CENTRO DE EDUCACAO ABERTA E A DISTANCIA            9  100.000000   \n",
            "81     CENTRO DE EDUCACAO ABERTA E A DISTANCIA            9   95.000000   \n",
            "82     CENTRO DE EDUCACAO ABERTA E A DISTANCIA            9   70.000000   \n",
            "83     CENTRO DE EDUCACAO ABERTA E A DISTANCIA            9   90.000000   \n",
            "...                                        ...          ...         ...   \n",
            "23814                       ESCOLA DE MEDICINA            8  100.000000   \n",
            "23816                       ESCOLA DE MEDICINA            8    0.000000   \n",
            "23817                       ESCOLA DE MEDICINA            8    0.000000   \n",
            "23818                       ESCOLA DE MEDICINA            8   71.428571   \n",
            "23819                       ESCOLA DE MEDICINA            8   97.435897   \n",
            "\n",
            "       Cluster  \n",
            "79           2  \n",
            "80           2  \n",
            "81           2  \n",
            "82           2  \n",
            "83           2  \n",
            "...        ...  \n",
            "23814        2  \n",
            "23816        2  \n",
            "23817        2  \n",
            "23818        2  \n",
            "23819        2  \n",
            "\n",
            "[7229 rows x 4 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Exibir os dados por cluster\n",
        "print(merged_data.groupby('Cluster').mean())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IMB9bCkotyYS",
        "outputId": "e7ae901e-3043-42db-ebe1-0b7f4274b66c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "         TotalEvasao  Desempenho\n",
            "Cluster                         \n",
            "0           3.672101   41.836689\n",
            "1           2.791315   82.359736\n",
            "2          11.808411   75.646969\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-20-5830d8304a9e>:2: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.\n",
            "  print(merged_data.groupby('Cluster').mean())\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Regressão**"
      ],
      "metadata": {
        "id": "roXwotRtMQPt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Selecionar apenas as colunas relevantes para a análise de regressão\n",
        "regression_data = merged_data[['TotalEvasao', 'Desempenho']].copy()\n",
        "\n",
        "X_reg = regression_data.drop(['TotalEvasao'], axis=1)\n",
        "y_reg = regression_data['TotalEvasao']\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)"
      ],
      "metadata": {
        "id": "J82uZx99pfCg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Criar um imputador para preencher os valores ausentes\n",
        "imputer = SimpleImputer(strategy='mean')\n",
        "\n",
        "# Aplicar o imputador aos dados de treinamento\n",
        "X_train_imputed = imputer.fit_transform(X_train)\n",
        "\n",
        "# Criar o modelo de regressão linear\n",
        "regressor = LinearRegression()\n",
        "\n",
        "# Treinar o modelo\n",
        "regressor.fit(X_train_imputed, y_train)\n",
        "\n",
        "# Aplicar o imputador aos dados de teste\n",
        "X_test_imputed = imputer.transform(X_test)\n",
        "\n",
        "# Fazer previsões no conjunto de teste\n",
        "y_pred = regressor.predict(X_test_imputed)\n",
        "\n",
        "# Calcular o erro médio quadrado\n",
        "mse = mean_squared_error(y_test, y_pred)\n",
        "print(f'Mean Squared Error: {mse}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PK5visg1xEkC",
        "outputId": "be4918a7-2dd1-428d-b05e-d58790f09be2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mean Squared Error: 20.903898432222675\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Validação** **cruzada**"
      ],
      "metadata": {
        "id": "fYxJxLVzMrom"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Validação cruzada para reduzir a chance de overfitting\n",
        "\n",
        "# Criar um imputador para preencher os valores ausentes\n",
        "imputer = SimpleImputer(strategy='mean')\n",
        "\n",
        "# Aplicar o imputador aos dados de treinamento\n",
        "X_train_imputed = imputer.fit_transform(X_train)\n",
        "\n",
        "# Criar o modelo de regressão linear\n",
        "regressor = LinearRegression()\n",
        "\n",
        "# Calcular a validação cruzada para o modelo\n",
        "cv_scores = cross_val_score(regressor, X_train_imputed, y_train, cv=5, scoring='neg_mean_squared_error')\n",
        "\n",
        "# Converter as pontuações negativas em positivas\n",
        "cv_scores = -cv_scores\n",
        "\n",
        "# Calcular a média e o desvio padrão das pontuações\n",
        "mean_cv_score = cv_scores.mean()\n",
        "std_cv_score = cv_scores.std()\n",
        "\n",
        "# Um valor mais baixo indica que as previsões do modelo estão mais próximas dos valores reais\n",
        "print(f'Cross-Validation Mean MSE: {mean_cv_score}')\n",
        "# Desvio padrão das pontuações de MSE - Dispersão dos valores em relação a média\n",
        "print(f'Cross-Validation Standard Deviation: {std_cv_score}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RpsIm7Vn2PfC",
        "outputId": "e10a951b-4de3-4e61-b350-424f3a66567c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cross-Validation Mean MSE: 21.434815108614583\n",
            "Cross-Validation Standard Deviation: 0.32516121513346635\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Random Forest\n",
        "\n",
        "# Criar um imputador para preencher os valores ausentes\n",
        "imputer = SimpleImputer(strategy='mean')\n",
        "\n",
        "# Aplicar o imputador aos dados de treinamento\n",
        "X_train_imputed = imputer.fit_transform(X_train)\n",
        "\n",
        "# Criar o modelo Random Forest\n",
        "rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)\n",
        "\n",
        "# Treinar o modelo\n",
        "rf_regressor.fit(X_train_imputed, y_train)\n",
        "\n",
        "# Obter a importância das características\n",
        "feature_importances = rf_regressor.feature_importances_\n",
        "\n",
        "# Imprimir a importância das características\n",
        "for feature, importance in zip(X_train.columns, feature_importances):\n",
        "    print(f'{feature}: {importance}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MnmtgcuTxRVf",
        "outputId": "f3d9bd4e-b968-4af6-a480-7a1168d5de9c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Desempenho: 1.0\n"
          ]
        }
      ]
    }
  ]
}