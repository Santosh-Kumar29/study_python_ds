number = "3214444424789566"
number_list = list(number)

new_numer = []
for i in range(len(number_list) - 1):
    # print("number", number_list)
    # print("<<list of number>>>>", number_list[i])

    for j in range(len(number_list) - i - 1):
        print("number term", number_list[j])

        if number_list[i] > number_list[j + 1]:
            number_list[j], number_list[j + 1] = number_list[j + 1], number_list[j]
            print(">>>>all numbers is", number_list)
            # print(number_list)
new_numer.append(number_list)
print(new_numer)
