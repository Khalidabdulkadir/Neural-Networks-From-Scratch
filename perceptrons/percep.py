inputs = [1, 0,1, 0, 0]
weights = [0.7, 0.6, 0.5, 0.3, 0.4]
sum = 0
threshold = 1.5
for i in range(len(inputs)):
    res = inputs[i] * weights[i]
    print(f"Input: {inputs[i]}, Weight: {weights[i]}, Result: {res}")
    sum += res
if sum >= threshold:
    print("yes, you can go to the party")
else:print("no, you cannot go to the party")

    
               