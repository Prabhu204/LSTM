"""
author: Prabhu

"""

text = "John likes the blue house at the end of the street"

print(list(text))

vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;. !?:’/"\|_@#$%ˆ&*̃+'-=<>()[]{}""")

print(vocabulary)
char_to_ix = {}
for char_ in vocabulary:
    print(char_)
    if char_ not in char_to_ix:
        char_to_ix[char_] = len(char_to_ix)


print(char_to_ix)