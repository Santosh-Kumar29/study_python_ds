n = 10
number = [0, 1]
for j in range(0, n - 2):
    data = number[j] + number[j + 1]
    print(data)
    number.append(data)
print(number)
