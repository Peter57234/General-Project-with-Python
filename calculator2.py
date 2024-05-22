from prettytable import PrettyTable

x = PrettyTable()
while True:
 a = int(input("First value: "))
 b = int(input("Second values: "))
 c = input("Select your operation(+, - ,*, /): ")
 menu = input("Do you want to open menu?: ")
 goodbye = input("Do you want to exit?: ")
 if 'no' in goodbye:
     pass     
 if 'yes' in goodbye:
     break     
 if 'no' in menu:
     pass
 if 'yes' in menu:
  print("1. Language")
  print("2. Exit")
  print("3. Table")
  opt = input("Choose an option: ")
  if '3' in opt:
     x.add_column("Column 1", [])
     x.add_column("Column 2", [])
     x.add_column("Column 3", [])
     print(x)
  if '1' in opt:
      print("1. Vietnamese")
      print("2. English")
      ans = input("Choose an option: ")
      if '1' in ans:
        print("Đã lưu!")     
        while True:
         a = int(input("Số đầu tiên: "))
         b = int(input("Số thứ hai: "))
         c = input("Chọn dấu(+, -, x, /): ")
         menu = input("Bạn có muốn mở menu không?: ")
         goodbye = input("Bạn có muốn thoát khỏi chương trình không?: ")
         if 'no' in goodbye:
           print("")
         if 'yes' in goodbye:
           break        
         if 'no' in menu:
            print("")
         if 'yes' in menu:
            print("1. Ngôn ngữ")
            print("2. Thoát")
            print("3. Bảng")
            opt = input("Chọn một lựa chọn: ")
            if '2' in opt:
              print("")
            if '3' in opt:
              x.add_column("Column 1", [])
              x.add_column("Column 2", [])
              x.add_column("Column 3", [])  
            if '1' in opt:
               print("1. Vietnamese")
               print("2. English")
               ans = input("Chọn một lựa chọn: ") 
            if '2' in ans:
              print("Saved!")
              while True: 
               a = int(input("First value: "))
               b = int(input("Second values: "))
               c = input("Select your operation(+,-,*,/): ")
               menu = input("Do you want to open menu?: ")
               goodbye = input("Do you want to exit?: ")
               if 'no' in goodbye:
                pass     
               if 'yes' in goodbye:
                 break     
               if 'no' in menu:
                 pass
               if 'yes' in menu:
                 print("1. Language")
                 print("2. Exit")
                 print("3. Table")
                 opt = input("Choose an option: ")
                 if '2' in opt:
                   print("")
                 if '3' in opt:
                   x.add_column("Column 1", [])
                   x.add_column("Column 2", [])
                   x.add_column("Column 3", [])  
                 if '1' in opt:
                  print("1. Vietnamese")
                  print("2. English")
                  ans = input("Choose an option: ")
                  if '1' in ans:
                    print("Đã lưu!")
                    while True:
                     a = int(input("Số đầu tiên: "))
                     b = int(input("Số thứ hai: "))
                     c = input("Chọn dấu(+, -, x, /): ")
                     menu = input("Bạn có muốn mở menu không?: ")
                     goodbye = input("Bạn có muốn thoát khỏi chương trình không?: ") 
                     if 'no' in goodbye:
                      print("")
                     if 'yes' in goodbye:
                      break        
                     if 'no' in menu:
                      print("")
                     if 'yes' in menu:
                      print("1. Ngôn ngữ")
                      print("2. Thoát")
                      print("3. Bảng")
                      opt = input("Chọn một lựa chọn: ")
                      if '2' in opt:
                        print("")
                      if '3' in opt:
                        x.add_column("Column 1", [])
                        x.add_column("Column 2", [])
                        x.add_column("Column 3", [])  
                      if '1' in opt:
                        print("1. Vietnamese")
                        print("2. English")
                        ans = input("Chọn 1 lựa chọn: ")
                        if '2' in ans:
                          print("Saved!")
         if'+' in c:
           print("Bằng: ", a + b or b + a )
         if '-' in c :
           print("Bằng: ", a - b or b - a)
         if '*' in c:
           print("Bằng: ", a * b or b * a)
         try:       
          if '/' in c:
           print("Bằng: ", a / b or b / a)
         except:
            print("Lỗi!")    
      if '2' in ans:
        print("Saved!")
        a = int(input("First value: "))
        b = int(input("Second values: "))
        c = input("Select your operation(+,-,*,/): ")
          
      if '3' in opt:
         x.add_column("", [input])
         x.add_column("", [input])
         x.add_column("", [input])
      else:
         print("Error!")                       
      if '2' in opt:
          pass
 if '+' in c:
     print("Equal to: ", a + b or b + a )
 if '-' in c :
     print("Equal to: ", a - b or b - a)
 if '*' in c:
     print("Equal to: ", a * b or b * a)
 try:
  if '/' in c:
      print("Equal to: ", a / b or b / a)
 except:
  print("Error!")




