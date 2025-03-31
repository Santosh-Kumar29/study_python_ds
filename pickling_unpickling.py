import pickle

data = [{'name': 'John', 'age': 25, 'gender': 'Male'}]
# pickling write binary, read binary
with open("data", 'wb') as f:
    pickle.dump(data, f)
print(">>>", pickle)
with open('data', 'rb') as f:
    new_data = pickle.load(f)
print(new_data)