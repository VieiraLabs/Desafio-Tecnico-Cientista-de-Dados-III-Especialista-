# Desafio Técnico — Cientista de Dados III (Especialista)

## Contexto

Este desafio simula um cenário real de desenvolvimento de soluções de dados e inteligência artificial aplicadas ao contexto de Recursos Humanos.

## Objetivo

Construir uma aplicação de People Analytics com IA que seja capaz de:

1. prever risco de saída de colaboradores
2. gerar insights automatizados com apoio de LLM
3. permitir interação por meio de um agente
4. demonstrar organização técnica, visão arquitetural e capacidade de liderança

## Dados

Utilize dados públicos.

Sugestões:

- IBM HR Analytics Employee Attrition Dataset
- outro dataset público compatível com o problema proposto

Caso opte por outro dataset, explique a escolha no README.

## Escopo do Projeto

### 1. Machine Learning

Desenvolver um modelo preditivo de churn de colaboradores.

#### Requisitos mínimos

- definir claramente a variável alvo
- realizar análise exploratória dos dados
- fazer tratamento e preparação dos dados
- realizar feature engineering relevante
- treinar pelo menos dois modelos diferentes
- comparar os resultados
- justificar a escolha do modelo final

#### Métricas esperadas

Utilize métricas adequadas ao problema, como por exemplo:

- ROC-AUC
- Precision
- Recall
- F1-score

#### Explicabilidade

Apresente a explicabilidade do modelo por meio de uma ou mais abordagens, como:

- feature importance
- SHAP
- análise dos principais fatores associados ao risco de saída

O objetivo não é apenas prever, mas também explicar de forma clara os fatores que influenciam o resultado.

### 2. LLM

Implementar uma camada de inteligência com LLM para geração de insights a partir dos resultados do modelo.

O sistema deve ser capaz de responder perguntas como:

- por que este colaborador apresenta alto risco de saída?
- quais fatores mais contribuíram para esse risco?
- quais ações podem ser sugeridas para retenção?

#### Requisitos mínimos

- uso de prompt estruturado
- utilização de contexto vindo do modelo preditivo
- resposta organizada e consistente
- output estruturado em JSON ou formato equivalente

#### Exemplo de saída esperada

    {
      "risk_level": "high",
      "main_factors": ["baixo salário", "tempo sem promoção"],
      "recommended_actions": ["revisão salarial", "plano de carreira"]
    }

Não é esperado apenas um chat genérico. O LLM deve estar conectado ao problema, aos dados e à lógica do sistema.

### 3. Orquestração / Agent

Construir um fluxo simples de agente que orquestre as etapas da solução.

Esse agente deve ser capaz de:

- receber uma pergunta do usuário
- decidir quais ações precisam ser executadas
- consultar dados necessários
- acionar o modelo preditivo quando necessário
- gerar a resposta final com apoio do LLM

#### Exemplos de fluxo

Exemplo 1:

- o usuário pergunta qual colaborador apresenta maior risco
- o agente consulta os dados
- o agente executa ou consulta o resultado do modelo
- o agente responde com apoio do LLM

Exemplo 2:

- o usuário pergunta por que um colaborador específico está em risco
- o agente recupera features e score
- o agente monta o contexto
- o LLM gera a explicação final

#### Tecnologias sugeridas

Você pode utilizar uma das abordagens abaixo:

- OpenAI Agents SDK
- LangChain
- LlamaIndex
- implementação própria

O foco principal é demonstrar capacidade de orquestração, desenho lógico e clareza técnica.

### 4. Interface (Frontend) (Ponto Extra - Não é obrigatório)

Construir uma interface simples para interação com o sistema.

#### Requisitos mínimos

- visualização do risco de colaboradores
- listagem ou consulta de colaboradores
- tela ou área para interação com o agente

#### Tecnologias sugeridas

- Next.js com TypeScript
- Streamlit para uma versão simplificada

#### Diferenciais

Serão considerados diferenciais positivos:

- gráficos e indicadores
- filtros
- navegação clara
- boa experiência de uso
- organização visual coerente com um produto real

O frontend não precisa ser sofisticado, mas deve permitir demonstrar a aplicação funcionando de ponta a ponta.

### 5. Arquitetura

Descrever a arquitetura da solução de forma clara.

Essa descrição pode ser feita por meio de:

- diagrama
- documento em Markdown
- combinação dos dois

#### Espera-se que a arquitetura contemple

- camada de dados
- processamento e preparação
- treinamento do modelo
- serviço de inferência
- camada de agente
- integração com LLM
- interface do usuário

### 6. Backend / Produção

Não é obrigatório ter uma solução completa em produção, mas é importante demonstrar maturidade de engenharia.

#### Diferenciais recomendados

- API para inferência, por exemplo com FastAPI
- organização modular do projeto
- separação entre treinamento, inferência e interface
- uso de Docker
- versionamento de artefatos
- preocupação com reprodutibilidade

### 7. Liderança Técnica e Mentoria

Além da entrega técnica, o candidato deve responder em um documento separado como conduziria tecnicamente a evolução dessa iniciativa dentro de um time.

#### O documento deve responder aos seguintes pontos

##### a) Estrutura do time

Explique como você organizaria o time para desenvolver e evoluir essa solução.

Considere, por exemplo:

- Cientista de Dados I
- Cientista de Dados II
- Engenheiro de Dados ou de Machine Learning
- QA
- Produto
- Design, se fizer sentido

##### b) Mentoria

Explique como você apoiaria o desenvolvimento técnico de profissionais mais juniores.

Esperam-se exemplos práticos, como:

- code review
- pair programming
- definição de trilhas de desenvolvimento
- acompanhamento técnico
- sessões de estudo
- definição de boas práticas

##### c) Qualidade e evolução contínua

Explique como garantiria qualidade e sustentabilidade da solução ao longo do tempo.

Considere pontos como:

- testes
- monitoramento
- controle de drift
- documentação
- governança
- critérios de evolução técnica

## Entregáveis

A entrega deve conter, no mínimo:

- repositório com o código-fonte
- README com instruções de execução
- README das decisões técnicas
- descrição da arquitetura

## Observações

- evitar uma solução restrita apenas a notebook
- evitar uso superficial de LLM, sem contexto ou sem integração real com o problema
- priorizar clareza, consistência e tomada de decisão
- a solução não precisa estar perfeita visualmente, mas deve demonstrar maturidade técnica
- mais importante do que complexidade é a capacidade de justificar decisões

## Dicas

- pense na solução como produto, não apenas como exercício técnico
- priorize interpretabilidade além de performance
- use o LLM como camada de explicação e apoio à decisão
- demonstre organização de código e visão arquitetural
- mostre como você estruturaria o trabalho para evoluir com um time, e não apenas como faria sozinho
