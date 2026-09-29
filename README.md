# Revisão de Literatura: O Efeito de Técnicas de Aprimoramento de Imagens sobre Redes Neurais Convolucionais

Esse repositório consolida os artefatos de pesquisa para o estudo sistemático sobre como técnicas clássicas de pré-processamento de imagens (como redução de ruído, equalização de histograma e suavização) afetam o desempenho de classificadores baseados em Redes Neurais Convolucionais (CNNs). 

## Contexto e Hipótese
Em *pipelines* de visão computacional, assume-se frequentemente que uma imagem com melhor qualidade perceptual (para o olho humano) resultará em melhor desempenho de classificação. No entanto, a literatura tem demonstrado que o oposto ocorre com notável frequência. 

A hipótese central deste trabalho é que as CNNs utilizam componentes espectrais de alta frequência como sinais discriminativos primários. Como muitas técnicas de aprimoramento atuam justamente atenuando ou modificando essas frequências — muitas vezes para suavizar ruídos e texturas indesejadas aos olhos humanos — elas inadvertidamente removem as características que a rede neural utiliza para a tomada de decisão, causando uma queda sistêmica na acurácia do modelo.

