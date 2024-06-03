# Classificação de Imagens (CIFAR-10)

Sobre a ponderada, criei uma rede neural como conforme abaixo:

```
model = Sequential([
                    Conv2D(32, (2,2), activation="relu", input_shape=(32, 32, 3)),
                    MaxPooling2D(2),
                    Conv2D(32, (2,2), activation="relu"),
                    MaxPooling2D(2),
                    Flatten(),
                    Dense(32, activation="relu"),
                    Dense(10, activation="softmax")
])
```

Após o treino, o modelo teve uma acurácia de ~0.67.

![treino](/imgs/train.png)

O modelo foi salvo em um arquivo `.h5` e a API puxa-o para dar as predições.

A API retorna o seguinte JSON:

`prediction:` Mensagem onde a rede neural responde qual objeto ela tem a maior convicção da imagem ser.

`certainty:` Valor em string da porcentagem de convicção normalizado entre 0 e 1.

### Caso 1: Imagem aleatória de barco naval antigo

![case1](/imgs/case1.png)

`prediction:` "This is a ship" (correto)

`certainty`: 0.99

### Caso 2: Imagem do meu cachorro

![case2](/imgs/case2.png)

`prediction:` "This is a bird" (errado)

`certainty`: 0.34

### Case 3: Imagem de um veado na vida real

![case3](/imgs/case3.png)

`prediction:` "This is a deer" (correto)

`certainty`: 0.91

## Resultados finais

O modelo possui uma acurácia mediana, pois ela é básica somente para esta ponderada. O modelo possui dificuldades para acertar imagens com algum corte ou o ângulo de captura diferente, como visto no caso do meu cachorro (Caso 2), por exemplo.