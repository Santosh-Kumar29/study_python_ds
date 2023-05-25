# linear search
# number = [1, 2, 4, 3, 44, 3]
# target = 2
# for i in range(len(number) - 1):
#     if number[i] == target:
#         print(i)
#         break
# else:
#     print("not found")
# test
# number = [1, 2, 3, 2, 4, 3]
# target = 1
# data = []
# for i in range(len(number)-1):
#     data = number[i] + number[i + 1]
#     if data == target:
#         print(i, i+1)
#         break
# else:
#     print("not found")


# Binary search
# binary search is an searching algorithm which is used to find a specific target value within a sorted list of item
number = [1, 2, 4, 3, 44, 3]
target = 4
low = 0
high = len(number) - 1
while low <= high:
    mid = (low + high) // 2
    if number[mid] == target:
        print(mid)
    elif number[mid] < target:
        low = mid + 1
    else:
        high = mid -1
        break
print("-1")
