#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def somatorio(num):
    lista = list(range(1, num+1))
    print(sum(lista))


n = int(input("Digite o valor de 'N':"))
somatorio(n)


# In[ ]:


def qtdeNumerosParesImpares(lista):
    pares = 0
    impares = 0
    for i in lista:
        if i % 2 == 0:
            pares = pares + 1
        else:
            impares = impares + 1
    return pares, impares


lista = []
for i in range(10):
    num = int(input("Digite o nÃºmero %d:" % (i+1)))
    lista.append(num)


pares, impares = qtdeNumerosParesImpares(lista)
print("Quantidade numeros pares %d" % pares)
print("Quantidade numeros Ã­mpares %d" % impares)


# In[ ]:


def somarPares():
    qnt = input("Digite um número inteiro: ")

    soma = 0
    for n in range(0, int(qnt), 2):
        soma = soma + n
        print(n)
    return soma

s = somarPares()
print(s)


# In[ ]:


def verificaNota():
    nota = 0
    while nota < 7:

        nota = float(input('digite a nota do aluno: '))

        if nota < 7:
            print('Reprovado!')

        else:
            print('Aprovado')
            break
    return nota

v = verificaNota()
print(v)

