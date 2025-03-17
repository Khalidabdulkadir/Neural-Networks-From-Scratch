threshold = 1.5
inputVals = [1,0,1,1,0]
weights = [0.2,0.4,0.6,0.8,0.5]
sum = 0

for i in range(len(inputVals)):
    multiplier = inputVals[i] * weights[i]
    sum += multiplier
if sum >= threshold:
    print("Then i can Go out")
else:
    print("Not Going Out")