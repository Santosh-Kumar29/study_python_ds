# char = "Hello, This is me santosh"
# vowel = [i for i in char if i in "aeiou"]
# print(vowel)


# number = [1.23, 2.34, 10, -22.11, 4, 222, -44]
# number = [i if i > 0 else 0 for i in number]
# print(number)


# fruits = ['apple', 'aango', 'banana', 'cherry']
# find_data = [i for i, fruits in enumerate(fruits) if "a" in fruits]
# print(find_data)
# for adding indexing while using dict comprehension
# data = {f: i for i, f in enumerate(fruits)}
# print(data)

# for removing key and value
# remove_key = {'apple', 'cherry'}
# data = {key: fruits[key] for key in range(len(fruits)) if fruits[key] not in remove_key}
# print(data)

fruits = {
    'fruit1': 'apple',
    'fruit2': 'mango',
    'fruit3': 'banana',
    'fruit4': 'cherry'
}

fruits['fruit1'] = 'orange'
print(fruits)
# find_data = {}
# for key, value in fruits.items():
#     if "app" in value:
#         find_data[key] = value
# print(find_data)
