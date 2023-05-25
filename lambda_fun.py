# data = "skgovindoks"
# get_upper = lambda string: string.upper()
# print(get_upper(data))


# convert boolean in coorect format
boolean_char = lambda \
    input_str: True if input_str.strip().lower() == "true" else False if input_str.strip().lower() == "false" else None


print(boolean_char("false"))
