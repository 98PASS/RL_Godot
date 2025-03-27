# Deep Imitation Learning para Agentes Inteligentes: Implementação e Análise na Godot Engine

![image](https://github.com/user-attachments/assets/d178a3cd-a8b7-4d44-8379-452a5f16a828)  
*Imitation Learning com Godot RL Agents*

## Resumo

O aprendizado por imitação profunda desempenha um papel fundamental no treinamento de agentes inteligentes. Observa-se uma tendência crescente na utilização de demonstrações humanas para acelerar o processo de aprendizado em ambientes complexos. Visando a uma abordagem eficiente e escalável, este trabalho propõe a aplicação de deep imitation learning em ambientes desenvolvidos na Godot Engine, utilizando o framework Godot RL Agents. No método proposto, o agente é treinado por meio de comportamentos registrados por um especialista humano, captando padrões de ação em um ambiente de minijogo. Essa abordagem permite uma transferência direta de conhecimento, garantindo eficácia na execução de tarefas específicas. A arquitetura utilizada combina redes neurais com técnicas de aprendizado supervisionado, otimizando o processo de imitação. Para validar a proposta, um exemplo prático foi implementado, demonstrando a capacidade do agente de reproduzir comportamentos humanos com alta taxa de sucesso. Os resultados obtidos confirmam a viabilidade da abordagem proposta para o treinamento de agentes inteligentes em ambientes virtuais.

**Palavras-chave:** Deep Imitation Learning, Reinforcement Learning, Godot Engine, Godot RL Agents, Aprendizado por Imitação.

[![Godot](https://img.shields.io/badge/Godot-478CBF?style=flat&logo=godot-engine)](https://godotengine.org)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=flat&logo=python)](https://python.org)

## Introdução

O Aprendizado por Imitação (Imitation Learning) surge como uma abordagem inovadora no aprendizado de máquina, capacitando agentes a adquirirem habilidades através da observação e emulação de comportamentos exemplares. Diferentemente do Aprendizado por Reforço (Reinforcement Learning), que se fundamenta em funções de recompensa e exploração ambiental, esta técnica foca na reprodução direta de ações a partir de demonstrações pré-existentes.

Esta metodologia se revela particularmente vantajosa em domínios complexos, como o desenvolvimento de jogos, onde a especificação manual de funções de recompensa se mostra inviável ou subótima. Ao invés de tentar criar sistemas de recompensa complexos, ou grandes árvores de decisão, a imitação permite que o agente aprenda comportamentos complexos diretamente de exemplos demonstrados por um especialista humano ou artificial.

O cenário atual do Aprendizado por Imitação abrange diversas técnicas, incluindo o Behavioral Cloning e o Aprendizado Adversarial por Imitação (GAIL), cada qual com suas particularidades e aplicações. Contudo, desafios como a dependência da qualidade dos dados demonstrativos e a capacidade de generalização em ambientes dinâmicos ainda demandam atenção.

## 1. Abordagem Proposta

Este trabalho explora a aplicação do Aprendizado por Imitação na Godot Engine, visando treinar agentes virtuais em ambientes de minijogos a partir de demonstrações humanas. A integração do framework Godot RL Agents com bibliotecas especializadas de aprendizado por imitação permite superar os desafios técnicos associados à criação de comportamentos complexos em jogos, oferecendo uma alternativa eficiente aos métodos tradicionais de Aprendizado por Reforço.

A metodologia proposta envolve a coleta de dados de demonstração de um especialista humano jogando o minijogo na Godot Engine, o treinamento de um modelo de aprendizado por imitação utilizando esses dados, e a aplicação do modelo treinado para controlar o comportamento do agente virtual no jogo. Os passos para realizar o trabalho são:

* Coleta de dados de demonstração de um especialista humano jogando o minijogo na Godot Engine.
* Treinamento de um modelo de aprendizado por imitação utilizando esses dados.
* Aplicação do modelo treinado para controlar o comportamento do agente virtual no jogo.

A utilização da Godot Engine como plataforma de desenvolvimento oferece um ambiente flexível e acessível para a implementação e experimentação de técnicas de aprendizado por imitação em jogos.

## 2. Fundamentação Teórica

A área de Aprendizado por Imitação (Imitation Learning - IL) busca capacitar agentes a aprenderem políticas comportamentais a partir de demonstrações fornecidas por um expert. Diferentemente do Aprendizado por Reforço (Reinforcement Learning - RL), que se baseia em recompensas para guiar o aprendizado, o IL utiliza exemplos diretos de comportamento desejado. Essa abordagem se mostra particularmente útil em cenários onde a definição de funções de recompensa complexas ou a exploração de ambientes vastos se tornam desafiadoras.

### 2.1 Aprendizado por Comportamento (Behavioral Cloning - BC)

O BC é uma das técnicas mais simples e diretas de IL. Ele consiste em treinar um modelo para mapear diretamente as observações do ambiente às ações demonstradas pelo expert. Matematicamente, o objetivo do BC é minimizar a seguinte função de perda:

$$ L(θ) = -Σ log πθ(a|s) $$

onde π θ (a∣s) representa a política do agente parametrizada por θ, s é o estado observado, a é a ação demonstrada pelo expert e D expert é o conjunto de demonstrações do expert. O BC é eficaz quando as demonstrações cobrem uma ampla gama de estados e ações relevantes, mas pode sofrer com o problema de "desvio de distribuição" (distributional shift) quando o agente encontra estados fora da distribuição de treinamento.

### 2.2 Aprendizado Adversarial por Imitação (Generative Adversarial Imitation Learning - GAIL)

O GAIL, por sua vez, adota uma abordagem adversarial para o aprendizado de políticas. Ele utiliza uma rede discriminadora para distinguir entre as ações do agente e as ações do expert, enquanto o agente tenta enganar o discriminador gerando ações que se assemelham às do expert. O GAIL formula o problema de aprendizado como um jogo minimax entre o agente e o discriminador. A função de perda do GAIL pode ser expressa como:

$$ min_π max_D E_π[log D(s,a)] + E_π*[log(1-D(s,a))] $$

onde Dw​(s,a) representa a saída do discriminador parametrizado por W. O GAIL é capaz de aprender políticas complexas e robustas, mesmo em ambientes com demonstrações limitadas, pois não depende diretamente da função de recompensa do ambiente.

### 2.3 Stable Baselines3 (SB3)

Stable Baselines3 (SB3) é uma biblioteca Python que fornece implementações de algoritmos de RL e IL, como PPO (Proximal Policy Optimization), que foi utilizado neste estudo. O PPO é um algoritmo de RL on-policy que otimiza uma função objetivo surrogate para evitar grandes atualizações de política que poderiam desestabilizar o treinamento. O SB3 também oferece suporte a técnicas de IL, como o BC, facilitando a implementação e experimentação com diferentes abordagens de aprendizado.

## 3. Godot Engine e o Ambiente de Demonstração

### 3.1 Godot Engine

A Godot Engine é um motor de jogo 2D e 3D de código aberto e gratuito, conhecido por sua flexibilidade e facilidade de uso. Sua arquitetura baseada em nós permite a criação de jogos complexos de forma modular e intuitiva. A Godot suporta várias linguagens de script, incluindo GDScript (sua linguagem nativa) e C#, o que a torna versátil para diferentes tipos de projetos.

![image](https://github.com/user-attachments/assets/48eb5ab6-3f74-4c20-a717-952b7fb4c96e)

### 3.2 Ambiente de Demonstração

O ambiente de demonstração utilizado neste estudo, é um cenário simples projetado para ilustrar os conceitos de IL. Nele, um robô precisa navegar por um cenário para alcançar um objetivo. O ambiente fornece observações sensoriais ao robô e permite que ele execute ações para se mover. A complexidade do ambiente pode ser ajustada para explorar diferentes aspectos do aprendizado por imitação.

![image](https://github.com/user-attachments/assets/50f0aaa5-6e8c-4206-bfd4-3243425de649)

## 4. Metodologia

### 4.1 Preparação do Ambiente: Guia Passo a Passo

Para reproduzir este experimento, siga os passos abaixo para configurar o ambiente:

1.  **Instalação da Godot Engine (.NET):**

    * Baixe a versão .NET da Godot Engine no seguinte link: [https://godotengine.org/download/windows/](https://godotengine.org/download/windows/).
    * Extraia o arquivo baixado e execute o executável da Godot Engine.

2.  **Criação do Ambiente Virtual Python:**

    * Abra o terminal (prompt de comando no Windows ou terminal no macOS/Linux).
    * Navegue até o diretório onde você deseja criar o ambiente virtual.
    * Execute o seguinte comando para criar o ambiente virtual chamado "venv":

        ```bash
        python -m venv venv
        ```

    * Ative o ambiente virtual:
        * No Windows:

            ```bash
            venv\Scripts\activate
            ```

        * No macOS/Linux:

            ```bash
            source venv/bin/activate
            ```

3.  **Instalação das Bibliotecas Necessárias:**

    * Com o ambiente virtual ativado, execute os seguintes comandos para instalar as bibliotecas `godot-rl` e `imitation`:

        ```bash
        pip install godot-rl
        pip install imitation
        ```

4.  **Aquisição do Ambiente de Demonstração:**

    * Baixe o ambiente de demonstração no seguinte link: [https://huggingface.co/learn/deep-rl-course/unitbonus5/getting-started](https://huggingface.co/learn/deep-rl-course/unitbonus5/getting-started).
    * Extraia o arquivo baixado e coloque-o em um diretório de sua escolha.

5.  **Configuração do Script do Robô:**

    * Abra o projeto do ambiente de demonstração na Godot Engine.
    * Edite o script do robô para configurar os sensores, a função de recompensa e outras propriedades relevantes, conforme necessário para o experimento.

6.  **Coleta de Demonstrações do Expert:**

    * Execute a cena de demonstração na Godot Engine.
    * Jogue pelo menos 30 episódios para coletar dados de demonstração.
    * Exporte os dados de demonstração no arquivo `expert_demos.json` ao finalizar o jogo utilizando Alt+F4.

7.  **Exportação do Jogo na Godot Engine:**

    * Exporte o jogo na Godot Engine para criar um executável independente.

8.  **Gravação de Demos (Opcional):**

    * Grave as demos para visualização posterior, se desejar.

9.  **Download do Script de Treinamento:**

    * Baixe o script `sb3_imitation.py` no seguinte link: [https://github.com/edbeeching/godot_rl_agents/blob/main/examples/sb3_imitation.py](https://github.com/edbeeching/godot_rl_agents/blob/main/examples/sb3_imitation.py).

10. **Execução do Script de Treinamento:**

    * Abra o terminal no diretório onde você salvou o script `sb3_imitation.py`.
    * Execute o seguinte comando, substituindo `"path_to_ILTutorial_executable"` pelo caminho do executável do jogo exportado e `"path_to_expert_demos.json"` pelo caminho do arquivo `expert_demos.json`:

        ```bash
        sb3_imitation.py --env_path="path_to_ILTutorial_executable" --bc_epochs=100 --gail_timesteps=1450000 --demo_files "path_to_expert_demos.json" --n_parallel=4 --speedup=20 --onnx_export_path=model.onnx --experiment_name=ILTutorial
        ```

11. **Aplicação do Modelo Treinado:**

    * Após o treinamento, o modelo `model.onnx` será gerado.
    * Aplique o modelo `model.onnx` ao robô no ambiente da Godot Engine.

## 4.2 Implementação do Aprendizado por Imitação

O script `sb3_imitation.py` implementa o aprendizado por imitação utilizando as bibliotecas `imitation` e `stable-baselines3`. Os trechos de código mais importantes incluem:

**Carregamento das Demonstrações:**

```python
trajectories = []
for file_path in args.demo_files:
    with open(file_path, "r") as file:
        data = json.load(file)

    for traj in data:
        trajectories.append(
            imitation.data.rollout.types.Trajectory(
                obs=np.array(traj[0]),
                acts=np.array(traj[1]),
                infos=None,
                terminal=True,
            )
        )
```

**Inicialização do Ambiente e do Modelo:**

```python
env = SBGSingleObsEnv(
    env_path=args.env_path,
    show_window=args.viz,
    seed=args.seed,
    n_parallel=args.n_parallel,
    speedup=args.speedup,
    obs_key="obs",
)

env = VecMonitor(env)

learner = PPO(
    env=env,
    policy="MlpPolicy",
    # ... outros parâmetros ...
)
```

**Treinamento com BC e GAIL:**

```python
if args.bc_epochs > 0:
    bc_trainer = bc.BC(
        # ...
    )
    bc_trainer.train(n_epochs=args.bc_epochs)

if args.gail_timesteps > 0:
    gail_trainer = GAIL(
        # ...
    )
    gail_trainer.train(args.gail_timesteps)
```

## Considerações Finais

Este trabalho explorou a aplicação do Aprendizado por Imitação (Imitation Learning - IL) em ambientes de minijogos desenvolvidos na Godot Engine, demonstrando a viabilidade e eficácia da utilização de demonstrações humanas para o treinamento de agentes virtuais. A integração do framework Godot RL Agents com bibliotecas especializadas de IL permitiu superar os desafios técnicos associados à criação de comportamentos complexos em jogos, oferecendo uma alternativa eficiente aos métodos tradicionais de Aprendizado por Reforço (Reinforcement Learning - RL).

A abordagem proposta, que combina a flexibilidade da Godot Engine com as capacidades de aprendizado do IL, mostrou-se capaz de gerar agentes virtuais que reproduzem com alta fidelidade os comportamentos demonstrados por um especialista humano. A capacidade de aprender diretamente de exemplos práticos elimina a necessidade de definir funções de recompensa complexas, simplificando o processo de desenvolvimento e tornando-o acessível a desenvolvedores com diferentes níveis de experiência em aprendizado de máquina.

Os resultados obtidos neste estudo abrem portas para diversas aplicações futuras, como a criação de personagens não jogáveis (NPCs) com comportamentos mais realistas e adaptáveis, a geração automática de conteúdo de jogos e o desenvolvimento de ambientes de treinamento personalizados para robôs virtuais. Além disso, a metodologia proposta pode ser estendida para outros domínios além dos jogos, como simulações de treinamento em realidade virtual e aumentada.

## Referências

1.  **Godot RL Agents Documentation.** Disponível em: [https://github.com/edbeeching/godot_rl_agents](https://github.com/edbeeching/godot_rl_agents).

2.  **Imitation Learning with Godot RL Agents.** Disponível em: [https://huggingface.co/learn/deep-rl-course/unitbonus5/introduction](https://huggingface.co/learn/deep-rl-course/unitbonus5/introduction).

3.  **ENGINE, G. Download for Windows.** Disponível em: [https://godotengine.org/download/windows/](https://godotengine.org/download/windows/).

