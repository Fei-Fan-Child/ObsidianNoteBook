```python
file_name = input("enter the file name")  
  
try:  
    handle = open(file_name)  
  
    counts = {}  
    for line in handle:  
        if line.startswith("From"):  
            email_line = line.split()  
            email = email_line[1]  
            counts[email] = counts.get(email, 0) + 1  

    # 找出出现次数最多的邮箱
    bigemail = None  
    bigcounts = None  
    for email, count in counts.items():  
        if bigcounts is None or count > bigcounts:  
            bigemail = email  
            bigcounts = count  
  
    print(bigemail, bigcounts)  
  
except:  
    print("invalid input")
```