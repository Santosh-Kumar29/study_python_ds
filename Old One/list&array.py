# my_list  = [1, 2, 4, "sss", "2.33", [3, 5]]
# print(my_list)
# import array as arr
#
# my_array = arr.array('i', [1, 2, 4, 3])
# print(my_array)



# remove char
# print(char.replace("e", remove_char))
# char = "hello"
# remove_char = "e"
# add_char = "f"
# new = ""
# for c in char:
#     if c == remove_char:
#         new += add_char
#     else:
#         new += c
# print(new)

# List

# fruits = ["apple", "banana", "orange", "mango"]
# fruits[2] = "lemon"
# print(fruits[2])

# fruits = ("apple", "banana", "orange", "mango")
# print(len(fruits))
# print(fruits[2])

# student = {"name": "John", "age": 20, "grade": "A"}
# student["name"] = "kumar"
# print(len(student["name"]))
#
# table = 4
# for i in range(1, 11):
#     data = table * i
#     print(f" {4} * {i} = {data}")


# add the number with sending multiple argumatns
# number = [1, 4, 3, 2, 4, 2, 1, 9]
# max_num = number[0]
# print(max_num)
# for i in range(len(number)):
#     if number[i] > max_num:
#         print(number[i])
#         max_num = number[i]
# print(max_num)

# list slicing
string = "myworld"
reverse = ""
for char in string:
    # print(reverse)
    print(">>", char)
    reverse = char + reverse
print(reverse)