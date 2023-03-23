from net import Net
from scallers import min_max

import pandas as pd

""" Informações sobre atributos do conjunto de dados 

Gravidezes: Para expressar o número de gestações
Glicose: Para expressar o nível de glicose no sangue
BloodPressure: Para expressar a medição da pressão arterial
SkinThickness: Para expressar a espessura da pele
Insulina: Para expressar o nível de insulina no sangue
IMC: Para expressar o índice de massa corporal
DiabetesPedigreeFunction: Para expressar a porcentagem de Diabetes
Idade: Para expressar a idade
Resultado: Para expressar o resultado final 1 é Sim e 0 é Não
"""

data = pd.read_csv("./diabetes_dataset.csv")

results = data[["Outcome"]]
data = data.drop(["No", "Outcome"], axis=1)

print(data)


min_max(data, [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "Age"
])

n = Net([8, 6, 4, 1], 0.08)
n.load("diabetes")

for entry, expect in zip(data.values, results.values):
    n.fit(entry, expect)
    n.save("diabetes")
