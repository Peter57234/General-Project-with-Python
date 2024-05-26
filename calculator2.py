from prettytable import PrettyTable
import math
import os
from openai import OpenAI

client = OpenAI(
   api_key= os.environ.get("")
)

x = PrettyTable()


def create(language):
    annouce = {
        'en' : ["First value: ", "Second values: ", "Select your operation(+, -, *, /): ", "Do you want to open menu?:  ", "Do you want to exit?: "],
        'vi' : ["Số đầu tiên: ", "Số thứ hai: ", "Chọn dấu(+, -, *, /): ", "Bạn có muốn mở menu không?: ", "Bạn có muốn thoát khỏi chương trình không?:  "],
    } 
    
    return (input(annouce[language][i])for i in range(5))   
def men(language):
   menus = {
      'en' : ["1. Language", "2. Exit", "3. Table", "4. Convert decimals to binary", "5. Unit conversion", "6. Prime", "7. AI help"],
      'vi' : ["1. Ngôn ngữ", "2. Thoát", "3. Bảng", "4. Chuyển đổi số thập phân sang hệ nhị phân", "5. Chuyển đổi đơn vị đo", "6. Số nguyên tố", "7. AI trợ giúp"],

   } 
   for item in menus[language]:
     print(item)
   return (input("Chọn 1 lựa chọn: "))
   
def binary_convert():
  try:  
    ans = int(input("Nhập số: "))
    x = bin(ans)
    return print("Bằng: ", x)
  except Exception:
    print("Lỗi! Vui lòng nhập 1 con số hợp lệ") 
def convert_unit():
  try:  
    ans = float(input("Nhập số mét: "))
    
    converted_rates = {
        "mét sang kilômét": 0.001,
        "mét sang centimét": 100,
        "mét sang milimét": 1000,
    }
    
    print("Chọn loại chuyển đổi:")
    for i, conversion in enumerate(converted_rates.keys(), start=1):
        print(f"{i}. {conversion}")

    
    choice = int(input("Nhập số tương ứng với loại chuyển đổi: "))
    conversion_key = list(converted_rates.keys())[choice - 1]

   
    result = ans * converted_rates[conversion_key]
    print(f"Bằng: {result} {conversion_key.split()[-1]}")
    return
  except ValueError:
    print("Lỗi! Vui lòng nhập 1 con số hợp lệ")
  except IndexError:
    print("Lỗi! Vui lòng chọn 1 lựa chọn hợp lệ")  
def convert_unit_pound():
   try: 
    ans = float(input("Nhập số gam: "))
    
    converted_rates = {
        "gam đến kilogam": 0.001,
        "gam đến héc tô gam": 0.01,
        "gam đến đề ca gam": 0.1,
    }
    
    print("Chọn loại chuyển đổi:")
    for i, conversion in enumerate(converted_rates.keys(), start=1):
        print(f"{i}. {conversion}")

    
    choice = int(input("Nhập số tương ứng với loại chuyển đổi: "))
    conversion_key = list(converted_rates.keys())[choice - 1]

   
    result = ans * converted_rates[conversion_key]
    print(f"Bằng: {result} {conversion_key.split()[-1]}")
    return
   except ValueError:
     print("Lỗi! Vui lòng nhập 1 con số hợp lệ")
   except IndexError:
     print("Lỗi! Vui lòng chọn 1 lựa chọn hợp lệ")
     
def chat_with_gpt(prompt): 
 stream = client.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages = [{"role": "user", "content": prompt}],
    temperature = 0.5,
    max_tokens = 1024,
    top_p = 1,
    frequency_penalty = 1,
    presence_penalty = 1,
    stream = True,
)
 for chunk in stream:
    if chunk.choices[0].delta.content is not None:
     print(chunk.choices[0].delta.content, end="")       

def eratosthenes():
  while True:
   try:
    n = int(input("Nhập giới hạn: "))
    exit = input("Bạn có muốn exit không?")
    dd = [True]*n
    s = []
    for i in range(2, n):
     if dd[i] == True:
      for e in range(i*i,n,i):
        dd[e] = False
    for i in range(2,n):
     if dd[i] == True:
      s.append(i)
    print(s)
    if 'yes' in exit:
     break     
    return
   except ValueError:
     print("Lỗi! Vui lòng nhập 1 con số hợp lệ")

def prime_check():
  while True:
   try: 
    n = int(input("Nhập số: "))
    exit = input("Bạn có muốn exit không?")
    check = True
    if n<2:
     check = False
    for i in range(2,n//2+1):         
     if n%i == 0:
      check = False
    if check == True:
     print("Số đó là số nguyên tố")
    else:
     print("Không phải là số nguyên tố")
    if 'yes' in exit.lower():
     break 
    return
   except ValueError:
     print("Lỗi! Vui lòng nhập 1 con số hợp lệ ")         
def calculate(a,b,c):
 try: 
     if c == '+':
      return a+b
     if c == '-':
      return a-b
     if c == '*':
      return a*b 
     if c == '/':
      return a/b 
     if c == '**':
      return a ** b  
 except Exception as e:
    print(f"Lỗi: {e}") 
def main():
 language = 'en'
 while True:
  a, b, c, menu, goodbye = create(language)
  a,b = int(a), int(b)
  if 'yes' in goodbye:
    break
  if 'yes' in menu:
    opt = men(language)
    if '3' in opt:
     ans = input("").split(',')
     x.add_column("Column 1", ans)
     x.add_column("Column 2", ans)
     x.add_column("Column 3", ans)
     print(x)
    if '4' in opt:
       binary_convert()
    if '5' in opt:
      print("1. Độ dài")
      print("2. Khối lượng")
      ans = input("Chọn 1 lựa chọn: ")
      if '1' in ans:
        convert_unit()
      if '2' in ans:
        convert_unit_pound()
    if '6' in opt:
      print("1. Kiểm tra số nguyên tố")
      print("2. Tìm số nguyên tố")
      ans = input("Chọn 1 lựa chọn: ")
      if '1' in ans:
        prime_check()     
      if '2' in ans:
       eratosthenes()
    if '7' in opt:
      print("Đã lưu")
      while True:
        user_input = input("You: ")
        if user_input.lower() in ["tạm biệt", "bye", "thoát", "exit"]:
          break
        response = chat_with_gpt(user_input)
        if response is None:
          print("")
        else:
          print("ChatGPT: ", response)         
    elif '1' in opt:
     print("1. Vietnamese")
     print("2. English")
     ans = input("Chọn 1 lựa chọn: ")
     if '1' in ans:
      print("Đã lưu!") 
      language = 'vi'
     if '2' in ans:
      print("Đã lưu!")
      language = 'en'   
  equal = math(a,b,c)
  print(f"Bằng:  {equal}")  
if __name__ == "__main__" :
  main()
