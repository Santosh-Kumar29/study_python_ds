#Given an array, find the maximum sum of any subarray of size k.
arr = [1, 2, 3, 4, 5, 6]

k = 3
max_sum = float("-inf")
window_sum = sum(arr[:k])

print(window_sum)

max_num = window_sum

for i in range(k, len(arr)):
    window_sum += arr[i] - arr[i - k]
    max_sum = max(max_sum, window_sum)
    
print(max_sum)




keys = ['a','b','c','d','e']
values = [1,2,3,4,5]  


data = {k:v for (k,v) in zip(keys, values)}

print(data)


"""


WSGI (Web Server Gateway Interface) is a synchronous standard for Python web apps, handling one request per thread/process, 
making it ideal for traditional, simple, or CPU-bound applications. ASGI (Asynchronous Server Gateway Interface) is its modern successor, 
supporting asynchronous (non-blocking) programming, WebSockets, and high-concurrency for real-time applications. 


"""