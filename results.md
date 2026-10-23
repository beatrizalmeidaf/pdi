# Resultados

Esse arquivo reúne o que foi executado e os resultados obtidos até agora. A modelagem, as hipóteses e os métodos estão no [README.md](README.md). As figuras e tabelas citadas aqui ficam em [docs/resultados/](docs/resultados/), e os dados brutos em `results/` (não versionado).

> **Escopo.** Todos os números são de **CIFAR-10** (32×32 px), com **3 CNNs** pré-treinadas, rodando em **CPU**. Ainda não há dataset de 224 px nem Vision Transformer, o que exige GPU (ver [§10](#10-o-que-ainda-não-foi-feito)). Por isso, trate as conclusões como **preliminares**.

---

## Sumário

1. [O que rodou](#1-o-que-rodou)
2. [Validação das implementações (E0)](#2-validação-das-implementações-e0)
3. [Limiares por técnica e janela operacional (H1, H6)](#3-limiares-por-técnica-e-janela-operacional-h1-h6)
4. [Achados específicos](#4-achados-específicos)
5. [Previsibilidade espectral (H2, H5)](#5-previsibilidade-espectral-h2-h5)
6. [Fórmula IRE (H7)](#6-fórmula-ire-h7)
7. [Sensibilidade por banda — oráculo (E1, H3)](#7-sensibilidade-por-banda--oráculo-e1-h3)
8. [Domain shift × perda de informação (E4, H4)](#8-domain-shift--perda-de-informação-e4-h4)
9. [Validação perceptual: BRISQUE e estudo com pessoas](#9-validação-perceptual-brisque-e-estudo-com-pessoas)
10. [O que ainda não foi feito](#10-o-que-ainda-não-foi-feito)
11. [Como reproduzir](#11-como-reproduzir)

---

## 1. O que rodou

| Execução | Config | Dados | Etapas | Tempo (CPU) | Status |
|---|---|---|---|---|---|
| Testes unitários | — | sintéticos | técnicas × referência, quantificador espectral, limiares, IRE, kit humano, pipeline offline | ~10 s | ✅ 23 passam, 1 pulado* |
| `make verify` | `sweep.yaml` | 50 imagens CIFAR-10 | E0 | ~1 min | ✅ |
| `make offline` | `offline.yaml` | 48 imagens sintéticas + CNN sem pesos | E1, E2, E4, E3 | ~1 min | ✅ (checa que o pipeline roda; números sem significado) |
| `make smoke` | `smoke.yaml` | 64 imagens CIFAR-10, 2 modelos | E1, E2, E4, E3 | ~2 min | ✅ |
| **`make sweep`** | `sweep.yaml` | **2000 imagens CIFAR-10, 3 modelos, 106 doses + 27 filtros-oráculo** | E1 (11 min), E2 (54 min), E4 (14 min), E3 (17 min) | ~1h40 | ✅ **base deste relatório** |
| `make human-export` | `sweep.yaml` | 12 imagens × 74 condições | kit do estudo com pessoas | <1 min | ✅ gerado; **sem respostas coletadas ainda** |

\* O teste do pipeline completo é pulado em ambientes sem torchvision (Anaconda local). No Python 3.12, com torch, ele passa.

Classificadores (pesos `chenyaofo/pytorch-cifar-models`, `model.eval()`, sem nenhum ajuste):

| Modelo | Acurácia nas 2000 imagens originais |
|---|---|
| ResNet-20 | 92,85 % |
| ResNet-56 | 94,85 % |
| VGG-16-BN | 93,70 % |

Esses valores batem com os publicados para esses pesos (92,6 %, 94,4 % e 94,2 % no teste completo).

Durante as execuções, foram encontrados e corrigidos dois bugs que afetariam os resultados:
- o *backend* rápido do Wiener (SciPy) gerava bordas escuras e pixels pretos em regiões planas;
- o bilateral do OpenCV usa distância L1 entre cores, o que deixava o filtro ~3,5 dB mais fraco que o do zero.

Os resultados abaixo já são **posteriores** às correções.

---

## 2. Validação das implementações (E0)

Todas as técnicas foram implementadas do zero em NumPy e comparadas com OpenCV/PIL/SciPy em 50 imagens reais do CIFAR-10, em todas as doses da varredura ([verificacao_tecnicas.csv](docs/resultados/verificacao_tecnicas.csv)).

| Técnica | Maior diferença média (níveis de cinza, 0–255) | Menor PSNR do zero × referência |
|---|---|---|
| Gaussiano, *unsharp*, laplaciano, mediana, Wiener | 0,00–0,10 | ≥ 58 dB (idênticos ou quase) |
| CLAHE, CLAHE misturado, equalização | 0,33 | 52,9 dB |
| Bilateral | 0,39 | 47,4 dB |
| SR bicúbica, ESPCN, FSRCNN | 0,73–0,78 | 45,7 dB |
| JPEG 4:2:0 / 4:4:4 | 1,61–1,73 | 37,1 dB (DCT em ponto flutuante × inteira no libjpeg) |

O quantificador espectral também passa nos testes de propriedade:
- a identidade dá ΔE = 0;
- as bandas somam a energia total, e sem janela a energia total é igual à variância dos pixels (Parseval);
- senoides caem na banda certa;
- a decomposição D = A + P é exata.

---

## 3. Limiares por técnica e janela operacional (H1, H6)

Tabela completa: [thresholds.csv](docs/resultados/thresholds.csv). Valores = faixa entre os 3 modelos. **BMD 1 p.p.** = intensidade em que a acurácia cai 1 ponto; o **BMDL** (limite inferior do IC 90 %, bootstrap pareado com 1000 reamostragens) é a dose segura conservadora. **Objetivo visual** = intensidade em que a técnica cumpre o que promete (−10 % de ruído, +10 % de nitidez ou +10 % de contraste). **Janela** = [objetivo visual, BMDL].

| Técnica (parâmetro) | BMD 1 p.p. | BMDL | BMD 5 p.p. | Objetivo visual | Janela operacional |
|---|---|---|---|---|---|
| Gaussiano (σ) | 0,43–0,45 | 0,41–0,43 | 0,54–0,56 | 0,33 | ✅ **σ ∈ [0,33; 0,41–0,43]** nos 3 |
| *Unsharp* (amount, σ = 1) | 0,45–0,56 | 0,28–0,37 | 1,29–1,54 | 0,26 | ✅ **[0,26; 0,28–0,37]** nos 3 |
| CLAHE misturado (α, clip 2) | 0,42–0,62 | 0,27–0,44 | não cruza até α = 1 (−1,9 a −2,5 p.p. em α = 1) | 0,32 | ✅ **[0,32; 0,41–0,44]** em 2 de 3 (no VGG o BMDL = 0,27 fica antes) |
| CLAHE (clip) | já no menor clip (1): −1,4 a −2,1 p.p. | — | 3,4–4,1 | 1 | só no limite: o clip mínimo já custa > 1 p.p. |
| Equalização (α) | 0,19–0,20 | 0,10–0,14 | 0,64–0,70 | 0,16 | ❌ o BMDL vem antes do objetivo |
| Laplaciano (c) | 0,09–0,13 | 0,05–0,08 | 0,31–0,35 | 0,09 | ❌ |
| Bilateral (σ_s) | 0,88–1,11 | 0,73–0,91 | 2,0–2,7 | 1,89 | ❌ |
| Mediana (k) | k = 3 já custa 11,6–18,0 p.p. | — | — | k = 3 | ❌ (parâmetro discreto: o menor passo já passa) |
| Wiener (k) | k = 3 já custa 11,8–16,7 p.p. | — | — | k = 3 | ❌ (idem) |
| JPEG 4:2:0 (qualidade) | 97,6–100 (no VGG, já na qualidade 100) | — | 84–90 | — | — |
| JPEG 4:4:4 (qualidade) | 95,2–95,9 | 94,1–96,8 | 74–86 | — | — |
| SR bicúbica (escala) | < 1,05 (1,05 já custa 3,9–5,7 p.p.) | — | 1,04–1,14 | — | — |
| ESPCN / FSRCNN (escala) | ×2 já custa 33–42 p.p. | — | ×2 | — | — |

![Limiares ResNet-20](docs/resultados/thresholds_cifar10_cifar10_resnet20.png)
*Um painel por técnica. Azul = Δacc com IC 90 %; vermelho tracejado = −flip; linha vermelha = BMD 1 p.p., com faixa do IC; verde tracejado = dose do objetivo visual; faixa verde = janela operacional. Os outros modelos: [ResNet-56](docs/resultados/thresholds_cifar10_cifar10_resnet56.png), [VGG-16-BN](docs/resultados/thresholds_cifar10_cifar10_vgg16_bn.png).*

**Leituras principais**
- **H1 confirmada no CIFAR-10.** Todas as técnicas degradam a acurácia de forma monotônica com a dose, inclusive as que só *adicionam* alta frequência (aguçamento) ou *redistribuem* (equalização). Nenhuma técnica melhorou a acurácia com significância: **não houve hormese**. O maior ganho pontual foi +0,2 p.p. (*unsharp* 0,1; CLAHE misturado 0,02), com IC cruzando 0.
- **Os limiares são estreitos e quase iguais entre arquiteturas.** No gaussiano, o BMD fica entre 0,43 e 0,45 nos 3 modelos. Entre perder 1 p.p. (σ = 0,44) e perder 5 p.p. (σ = 0,55) há só 0,11 de σ.
- **H6 parcialmente confirmada.** Existe janela operacional para **suavização gaussiana leve, *unsharp* leve e CLAHE fraco** (misturado ao original). **Não existe** para equalização global, laplaciano, bilateral, mediana e Wiener: nelas, quando o objetivo visual é atingido, o classificador já perdeu ≥ 1 p.p.
- As janelas que existem são **estreitas**: σ ∈ [0,33; 0,42] no gaussiano e amount ∈ [0,26; 0,32] no *unsharp*.

![Dose-resposta](docs/resultados/dose_response_cifar10.png)

---

## 4. Achados específicos

**JPEG: em alta qualidade, a perda vem da cor, não da compressão.**

| Qualidade | 4:2:0 (padrão PIL/libjpeg) | 4:4:4 (sem subamostragem de cor) |
|---|---|---|
| 100 | −0,55 / −0,75 / −1,35 p.p. | +0,05 / −0,20 / −0,25 p.p. |
| 99 | −0,60 / −0,45 / −1,15 | −0,10 / −0,10 / −0,20 |
| 98 | −0,65 / −0,90 / −1,15 | 0,00 / −0,05 / −0,10 |
| 95 | −3,05 / −1,95 / −1,85 | −1,45 / −1,35 / −1,05 |

*(ResNet-20 / ResNet-56 / VGG-16-BN)*

Na qualidade 98, o erro de **cor** da versão 4:2:0 é ~10× o de luminância. A subamostragem de croma sozinha já muda decisões. Sem ela, o JPEG só passa de 1 p.p. por volta da qualidade 95–96.

**CLAHE: o clip é quantizado em imagens pequenas.** Em blocos de 16×16 px (CIFAR, grade 2×2), o clip efetivo é `max(int(clip × 256 / 256), 1)`. Ou seja, **todo clip < 2 gera exatamente a mesma imagem**, e a versão mais fraca possível já custa 1,4–2,1 p.p. Por isso a dose fraca de CLAHE foi feita misturando a saída com o original (α).

**Redimensionar é muito danoso em 32 px.** Reduzir 32 → 30 px e voltar (escala 1,05) já custa 3,9–5,7 p.p.; o limiar fica abaixo da menor escala possível. Na mesma escala de redução, a **SR aprendida é melhor que a bicúbica** para o classificador (FSRCNN > ESPCN > bicúbica). Isso é coerente com a QP4 da revisão: aprimorar ajuda quando a entrada está de fato degradada.

| Escala | Bicúbica | ESPCN | FSRCNN |
|---|---|---|---|
| ×2 | −40,3 a −47,2 p.p. | −37,0 a −42,2 | −33,4 a −36,0 |
| ×3 | −67,2 a −72,2 | −63,6 a −68,4 | −62,6 a −64,2 |
| ×4 | −73,9 a −80,4 | −73,1 a −74,8 | −70,0 a −70,9 |

**Parâmetros discretos.** Para mediana e Wiener, k = 3 é o menor passo e já custa 12–18 p.p. Nessas técnicas não existe "dose fraca" em 32 px.

---

## 5. Previsibilidade espectral (H2, H5)

Uma regressão *ridge* prevê Δacc de cada condição usando só as outras técnicas (*leave-one-technique-out*, LOTO). Ela compara a **dose nominal** (o que a literatura usa), métricas de pixel e o **ΔE medido por banda** (proposta). Tabela: [loto.csv](docs/resultados/loto.csv).

| Preditor | R² LOTO (ResNet-20 / ResNet-56 / VGG) |
|---|---|
| parâmetro nominal (`dose_rank`) | 0,22 / 0,21 / 0,19 |
| PSNR | 0,06 / 0,06 / 0,06 |
| SSIM | 0,57 / 0,54 / 0,58 |
| energia da perturbação por banda (Yin et al.) | 0,41 / 0,50 / 0,56 |
| ΔE por banda | 0,71 / 0,65 / 0,71 |
| ΔE por banda + coerência de fase | 0,77 / 0,76 / 0,81 |
| ΔE por banda + coerência + cor | **0,82 / 0,78 / 0,81** |

**Por imagem** (prever se a decisão daquela imagem muda; AUC, validação agrupada por técnica):

| Preditor | AUC (ResNet-20 / ResNet-56 / VGG) |
|---|---|
| PSNR | 0,70 / 0,70 / 0,70 |
| ΔE por banda | 0,77 / 0,77 / 0,79 |
| ΔE por banda + coerência | **0,84 / 0,86 / 0,86** |

- **H2 confirmada.** O ΔE medido prevê o dano em técnicas não vistas muito melhor que o parâmetro nominal (R² 0,2 → 0,8) e que PSNR/SSIM.
- **H5 confirmada.** A coerência de fase (estrutura) acrescenta poder explicativo: +0,06 a +0,10 de R² e +0,07 de AUC.

![LOTO](docs/resultados/loto_r2_cifar10.png)

---

## 6. Fórmula IRE (H7)

**IRE = Σ w_b D_b^γ + Σ c_b D^cor_b^γ + Σ u_B (1 − ρ_B)^γ** e **P(decisão mudar) = 1 − e^(−IRE)**.

- D_b = energia da perturbação de luminância no anel b; D^cor_b = idem na cor; 1 − ρ_B = perda de coerência (estrutura); pesos ≥ 0 por modelo.
- Como só uma predição que muda pode virar erro, **IRE ≤ −ln(1 − τ) garante queda de acurácia ≤ τ**.

**Desempenho prevendo cada técnica sem tê-la visto** ([ire_summary.csv](docs/resultados/ire_summary.csv)):

| Modelo | R² (flip por condição) | Spearman | R² calibrando **só nos filtros ideais** | γ estimado |
|---|---|---|---|---|
| ResNet-20 | **0,872** | 0,953 | 0,799 | ≈ 0,75–0,85 |
| ResNet-56 | **0,860** | 0,953 | 0,808 | |
| VGG-16-BN | **0,865** | 0,948 | 0,763 | |

![IRE previsto × observado](docs/resultados/ire_pred_vs_obs_cifar10.png)

**Como a fórmula chegou aqui** (R² LOTO, ResNet-20):

| Versão | R² | O que mudou |
|---|---|---|
| Σ w·D (só energia de luminância, linear) | 0,13 | — |
| + expoente γ | 0,23 | resposta não linear |
| + perda de coerência (1 − ρ) | 0,85 | **o salto**: estrutura/fase importa mais que energia |
| + cor (D^cor) | 0,87 | corrige o JPEG (previsto/observado foi de 0,15 para ≈ 1,0) |
| + separar amplitude e estrutura (D = A + P) e brilho médio | 0,88 | ganho marginal; **não** corrigiu a equalização |

**Onde ela erra** (previsto/observado do flip médio, os 3 modelos):
- ≈ 1 (0,9–1,2) em bilateral, JPEG, JPEG 4:4:4, SR, *unsharp*, Wiener e laplaciano;
- **superestima a equalização**: HE ≈ 2,1×, CLAHE ≈ 2,1–2,3×, CLAHE misturado ≈ 1,7×;
- **subestima o blur forte**: gaussiano ≈ 0,6×, por saturação perto de 80 % de flip.

**Limiar previsto × medido** (flip de 1 %, LOTO; [ire_threshold_predictions.csv](docs/resultados/ire_threshold_predictions.csv)):
- dentro do IC do medido em **40 %** dos casos;
- erro relativo mediano de ≈ 0–20 % em mediana, Wiener, SR bicúbica, bilateral, HE, CLAHE misturado e gaussiano;
- erro de ~2× no laplaciano e no *unsharp*.

**Uso sem rede (`make safe-dose`).**
- Gaussiano, ResNet-20: a fórmula marca σ ≤ 0,40 como seguro (P(flip) previsto < 1 %). O medido foi BMD-flip 0,34 e BMD-acurácia 0,45.
- *Unsharp*: marca amount ≤ 0,2; o medido foi BMD-flip 0,11 e BMD-acurácia 0,53.
- A garantia de acurácia se manteve nos dois casos; a de flip ficou ligeiramente otimista.

![Pesos IRE](docs/resultados/ire_weights_cifar10.png)

### Ajuste da fórmula para as técnicas de equalização

A fórmula superestima o dano da equalização em ~2×. Foram testadas duas correções, sempre com validação LOTO e sem rodar as redes de novo (reaproveitando as decisões salvas). Os números abaixo usam 500 imagens; a tabela completa está em [ire_tonal_experiment.csv](docs/resultados/ire_tonal_experiment.csv).

| Variante | R² (R20 / R56 / VGG) | HE | CLAHE | CLAHE mist. | *Unsharp* | Laplaciano |
|---|---|---|---|---|---|---|
| **atual** (D + cor + coerência) | **0,88 / 0,86 / 0,87** | 2,06 / 1,89 / 1,85 | 1,98 / 1,99 / 2,35 | 1,55 / 1,61 / 1,75 | 0,98 / 0,98 / 1,16 | 0,81 / 0,92 / 0,98 |
| separar amplitude × estrutura por banda (D = A + P) | 0,88 / 0,86 / 0,87 | ≈ igual | ≈ igual | ≈ igual | ≈ igual | ≈ igual |
| tirar a **curva tonal global** (y ≈ f(x)) e usar só o resíduo | 0,70 / 0,66 / 0,63 | **0,55 / 0,50 / 0,50** | 2,60 / 2,83 / 3,14 | 2,4–3,0 | 2,5–3,1 | 1,7–2,2 |
| atual + termo tonal separado | 0,88 / 0,86 / 0,87 | igual à atual (o ajuste dá peso ≈ 0 ao termo) | | | | |

*(colunas de técnica = previsto/observado do flip; 1 = perfeito)*

O que se conclui:
- **A superestimação da equalização global é explicada pela curva tonal.** Quando a mudança que uma tabela de cinza explica é descontada, o HE deixa de ser superestimado (até passa a ser subestimado). Os modelos são mais tolerantes a remapeamentos globais de intensidade do que a energia espectral sugere.
- **Mas a correção global piora o resto.** O CLAHE é uma curva **local** (por bloco), que a curva global não captura. No aguçamento, parte do efeito é absorvida pela curva e o dano restante fica superestimado.
- **Decisão: manter a fórmula atual.** Ela é a melhor no conjunto. O viés na equalização fica documentado como limitação.

Próximos candidatos:
1. Curva tonal **por bloco**, compatível com o CLAHE.
2. Usar o resíduo tonal **junto** com as features atuais, e não no lugar delas.

Na prática, enquanto o viés existir, a fórmula erra **para o lado seguro** nas técnicas de equalização: diz que a dose segura é menor do que realmente é.

---

## 7. Sensibilidade por banda — oráculo (E1, H3)

Filtros ideais: removem (ganho 0) ou dobram (ganho 2) **um** anel de frequência por vez.

| Centro do anel (ciclos/pixel) | Remover: Δacc (R20 / R56 / VGG) | Dobrar: Δacc |
|---|---|---|
| 0,03 | −4,1 / −3,7 / −3,8 | −4,5 / −3,9 / −4,1 |
| 0,16 | −5,1 / −4,3 / −3,4 | −6,2 / −5,4 / −5,1 |
| 0,28 | **−7,4 / −5,7** / −3,4 | −5,8 / −5,3 / −3,4 |
| 0,41 | −6,6 / **−6,2** / −3,6 | −6,0 / −5,0 / −3,0 |
| 0,60 (cantos) | −3,1 / −2,5 / −0,9 | −2,2 / −2,1 / −0,8 |

- As **ResNets** dependem mais da **média-alta frequência** (0,28–0,41 ciclos/pixel), embora quase não haja energia ali.
- A **VGG** é mais plana e bem menos sensível nos cantos.
- Mexer em qualquer banda isolada custa 3–7 p.p. em ambos os sentidos, removendo ou amplificando.
- Esse perfil, aprendido só com filtros sintéticos, já prevê as técnicas reais com R² 0,76–0,81 (§6). Isso dá suporte parcial a H3. Falta comparar com outra resolução e com ViT para testar a dependência de dataset e arquitetura.

![Sensibilidade por banda](docs/resultados/band_sensitivity_cifar10.png)

---

## 8. Domain shift × perda de informação (E4, H4)

Mesmo extrator pré-treinado; muda só o quanto o modelo pode se adaptar às imagens processadas. Média dos 3 modelos, 5000 imagens de treino ([adaptation_summary.csv](docs/resultados/adaptation_summary.csv)).

| Condição | Congelado | Re-estimar BatchNorm | BN nas originais (controle) | Sonda orig → proc | Sonda proc → proc | Sonda orig → orig (teto) |
|---|---|---|---|---|---|---|
| Gaussiano σ = 1 | 39,6 % | **83,0 %** | 91,1 % | 40,2 % | 64,3 % | 93,7 % |
| Wiener k = 5 | 72,7 % | 82,2 % | 91,1 % | 73,0 % | 79,2 % | 93,7 % |
| Mediana k = 3 | 79,1 % | 81,7 % | 91,1 % | 79,3 % | 83,0 % | 93,7 % |
| Laplaciano c = 0,5 | 84,9 % | 89,3 % | 91,1 % | 84,9 % | 87,0 % | 93,7 % |
| JPEG qualidade 70 | 81,7 % | 79,2 % | 91,1 % | 81,6 % | 83,1 % | 93,7 % |
| JPEG qualidade 30 | 66,8 % | 66,7 % | 91,1 % | 67,2 % | 71,9 % | 93,7 % |
| Bilateral σ_s = 2 | 89,6 % | 89,2 % | 91,1 % | 89,6 % | 90,2 % | 93,7 % |
| *Unsharp* amount = 1 | 90,8 % | 89,9 % | 91,1 % | 90,5 % | 91,2 % | 93,7 % |
| CLAHE clip = 2 | 91,5 % | 89,1 % | 91,1 % | 91,3 % | 91,3 % | 93,7 % |

**H4 confirmada em parte.**
- **Suavização é sobretudo domain shift.** Só re-estimar as estatísticas do BatchNorm, sem mexer em nenhum peso, leva o blur σ = 1 de 39,6 % para 83,0 %, perto do controle de 91,1 %. A informação discriminativa continua na imagem; o modelo é que não estava calibrado para ela.
- **JPEG forte é perda estrutural.** O BN não recupera nada (66,8 % → 66,7 %) e a sonda re-treinada pouco.
- **Aguçamento e CLAHE em doses moderadas** custam pouco e já estão perto do controle. O espaço para recuperar é pequeno.

As sondas lineares (64–512 features) recuperam menos que o ajuste do BN nas CNNs de CIFAR. O fine-tuning completo, que exigiria GPU, não foi feito.

![Adaptação](docs/resultados/adaptation_cifar10.png)

---

## 9. Validação perceptual: BRISQUE e estudo com pessoas

### BRISQUE

O BRISQUE (via `piq`, CPU, menor = melhor) foi calculado para todas as 2000 imagens em todas as 106 condições, com as imagens ampliadas para 128 px.

Checagem de sanidade antes do uso (300 imagens):
- **em 32 px nativo é instável**: σ = 2 não sai pior que σ = 1;
- **ampliado para 128 px**, piora em 100 % das imagens com blur σ = 1 ou σ = 2 (e piora mais com σ = 2) e em 97 % com JPEG qualidade 10. Melhora com *unsharp* e CLAHE.
- Limitação: ampliado, não detecta ruído aditivo, que não é uma das técnicas do estudo.

| Técnica | BRISQUE melhora ≥ 5 % em | BMD do classificador (1 p.p.) | Janela pelo BRISQUE? |
|---|---|---|---|
| *Unsharp* | amount ≈ 0,58 | 0,45–0,56 | ❌ (melhora só depois do limiar) |
| Laplaciano | c ≈ 0,094 | 0,09–0,13 (BMDL 0,05–0,08) | ❌ (no limite) |
| Equalização | α ≈ 0,69 | 0,19–0,20 | ❌ |
| CLAHE (clip) | clip ≈ 3,7 | < 1 | ❌ |
| CLAHE misturado | nunca chega a 5 % (melhor: −3,6 % em α = 1) | 0,42–0,62 | ❌ |
| Gaussiano | nunca melhora; **piora 5 % em σ ≈ 0,43** | 0,43–0,45 | ❌ |
| Mediana, Wiener, bilateral, JPEG, SR | nunca melhora | — | ❌ |

**Resultado central.** Medida por uma métrica perceptual independente, **a melhora visual só chega depois que o classificador já perdeu ≥ 1 p.p., ou exatamente no limite**. Em nenhuma técnica o BRISQUE melhora antes do BMDL. Na suavização, o BRISQUE e o classificador "quebram" quase no mesmo ponto (σ ≈ 0,43–0,44).

Há duas leituras complementares:
- as janelas do §3 dependem do critério de "objetivo visual" (proxies como nitidez e ruído);
- por um critério perceptual geral, **não há dose de aprimoramento que seja grátis para o classificador** em imagens limpas de 32 px. Isso confirma a pergunta de pesquisa.

### Estudo com pessoas

O kit está pronto em `results/sweep/human_study/` (gerado por `make human-export`):
- 12 imagens × 74 condições + pares de atenção = **900 pares**;
- `index.html` roda localmente (4 MB), cada participante responde ~120 pares e o navegador baixa um CSV.

**Ainda não há respostas.** Para ±10 p.p. de precisão por dose, são necessários ~35 participantes; com `--technique` dá para focar numa técnica e precisar de menos gente. Depois de coletar os CSVs em `human_study/respostas/`, rode `make human-analyze`. O código da análise foi testado com participantes simulados, incluindo a exclusão de quem falha nos pares de atenção.

---

## 10. O que ainda não foi feito

| Item | Por quê | Como fazer |
|---|---|---|
| Dataset de 224 px (Imagenette/ImageNet) e ViT | Testa H3 (banda crítica × resolução/arquitetura) e a hormese do CLAHE relatada na literatura. Inviável em CPU (~1,6 M inferências, ViT-B/16 incluso) | `make sweep DATASETS=imagenette` com GPU (Colab/servidor) |
| Coleta do estudo com pessoas | depende de participantes | `make human-export` → enviar → `make human-analyze` |
| Fine-tuning completo no E4 | exige GPU | `adaptation.finetune_epochs > 0` |
| Fórmula IRE nas técnicas de equalização (superestima ~2×; a correção global testada não serviu) | ver §6 | curva tonal por bloco; resíduo tonal junto às features atuais |
| E5: entradas realmente degradadas (CIFAR-10-C) | QP4; download de ~2,9 GB | `bash scripts/download_data.sh cifar10c` |
| Métodos e Resultados no artigo LaTeX | — | a partir deste arquivo |

---

## 11. Como reproduzir

```bash
make setup && make download-sr && make download-cifar   # ambiente e dados
make test && make verify                               # E0
make sweep                                             # E1 + E2 + E4 + E3 no CIFAR-10 (~1h40 em CPU)
make safe-dose T=gaussian M=cifar10_resnet20           # dose segura pela fórmula, sem rede
make human-export                                      # kit do estudo com pessoas
```

Relatório automático completo, com todas as tabelas: `results/sweep/analysis/summary.md`.
