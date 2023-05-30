# SET
fruits = {"apple", "banana", "orange"}
# fruits.add("grape")
# fruits.remove("banana")
# print(fruits)
# print("apple" in fruits)

# for replace
# remove_char = "apple"
# add_char = "ok"
# modified_fruits = set()
#
# for char in fruits:
#     modified = char.replace(remove_char, add_char)
#     modified_fruits.add(modified)
#
# print(modified_fruits)

# LIST
# number = [1, 4, 3, 3, 2, 5, 5]
# number[0] = 3
# number.append(5)
# number.remove(5)
# print(number)
# remove_num = 4
# add_num = 11
#
# for i in range(len(number)):
#     if number[i] == remove_num:
#         number[i] = add_num
# print(number)


# new_list = []
# for num in number:
#     if remove_num == num:
#         new_list.append(add_num)
#     else:
#         new_list.append(num)
#
# print(new_list)


# ARRAY
# import array

# number_array = array.array('i', [1, 4, 3, 2])
# print(type(number_array))
# print(number_array[1])

# add_num = 5
# remove_num = 1
# for i in range(len(number_array)):
#     if number_array[i] == remove_num:
#         number_array[i] = add_num
# print(number_array)


# TUPLE
number = (1, 4, 3)
# print(number[0])

# for assign numner
# new_number = number + (20,)
# print(new_number)
#
# remove_num = 4
# add_num = 100
# new_num = ()
# for i in number:
#     if i == remove_num:
#         new_num += (add_num,)
#     else:
#         new_num += (i,)
# print(">>", new_num)

# print(number)


# DICTIONARY
person = {"name": "John", "age": 30, "city": "New York"}
# print(person["name"])
# print(person.get("name"))
# person["age"] = 40
# print(person)