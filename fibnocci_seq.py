n = 9
first = 0
second = 1
series = [first, second]
for i in range(0, n - 2):
    num = series[i] + series[i + 1]
    series.append(num)
print(series)
