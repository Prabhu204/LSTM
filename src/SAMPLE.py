"""
author: Prabhu
email: prabhu.appalapuri@gmail.com
"""

sample_data = [
    ("John likes the blue house at the end of the street".split(), ["NNP", "V","DET","JJ","NN","IN","DET","NN","IN","DET","NN"]),
    ("Viet was crazy dude".split(), ["NNP", "V","JJ","NN"])
]

# print(x[0] for x in sample_data)

for touple_ in sample_data:
    for index, key in enumerate(touple_):
        if index %2 != 0:
            print (key)

    # for item in touple_:
    #     print(item)
    #     print ('BB')