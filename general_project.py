from prettytable import PrettyTable
import math 
import os
from openai import OpenAI
import time
from pytube import YouTube 
from pydub import AudioSegment
import pygame
import qrcode
import random
import asyncio
import secrets
import string
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import bcrypt


client = OpenAI(api_key=os.environ.get(""))

x = PrettyTable()

def login_and_register():
  def hash_password(password):
    # Generate a salt
    salt = bcrypt.gensalt()
    # Hash the password
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed
  def register(): 
    nickname = input("Nhập tên người dùng: ")
    password = input("Nhập mật khẩu: ")
    hashed_password = hash_password(password)
          
    with open('register_log.txt', 'w') as files:
       files.write(nickname + "\n")
       files.write(hashed_password.decode() + '\n')
    print("Đăng ký thành công") 
  def check_password(stored_password, provided_password):
    # Check the password with the hashed version
    return bcrypt.checkpw(provided_password.encode(), stored_password.encode())    
       
  def login():
    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
        nickname = input("Nhập tên người dùng: ")
        password = input("Nhập mật khẩu: ")
        
        with open('register_log.txt', 'r') as file:
            lines = file.readlines()
            valid = False
            
            # Chạy qua các dòng, kiểm tra nickname và mật khẩu
            for i in range(0, len(lines), 2):
                saved_nickname = lines[i].strip()
                saved_hashed_password = lines[i+1].strip()
                
                if nickname == saved_nickname and check_password(saved_hashed_password, password):
                    valid = True
                    break
                 
            
        if valid:
            print("Đăng nhập thành công!")
            break
        else:
            attempts += 1
            if attempts < max_attempts:
               print(f"Sai mật khẩu hoặc tên người dùng, bạn còn {max_attempts - attempts} lần thử nữa.")
            else:   
             print("Sai mật khẩu hoặc tên người dùng, vui lòng thử lại sau vài phút.")
             time.sleep(1000)
  def main():
     choice = input("Chọn 1 để đăng ký, 2 để đăng nhập: ")
     if choice == '1':
        register()
        login()
     if choice == '2':
        login()
  if __name__ == '__main__':
     main()                

def input_user(language):
    annouce = {
        'en' : ["First value: ", "Second values: ", "Select your operation(+, -, *, /): ", "Do you want to open menu?:  ", "Do you want to exit?: "],
        'vi' : ["Số đầu tiên: ", "Số thứ hai: ", "Chọn dấu(+, -, *, /): ", "Bạn có muốn mở menu không?: ", "Bạn có muốn thoát khỏi chương trình không?:  "],
    } 
    
    return (input(annouce[language][i])for i in range(5))   
def men(language):
   menus = {
      'en' : ["1. Language", "2. Exit", "3. Table", "4. Convert decimals to binary", "5. Unit conversion", "6. Prime", "7. AI help", "8. Music", "9. Quick Response Code", "10. Probability ", "11. Caro game", "12. Password Generator", "13. Banking", "14. Feedback", "15 Log out"],
      'vi' : ["1. Ngôn ngữ", "2. Thoát", "3. Bảng", "4. Chuyển đổi số thập phân sang hệ nhị phân", "5. Chuyển đổi đơn vị đo", "6. Số nguyên tố", "7. AI trợ giúp", "8. Nhạc", "9. Mã phản hồi nhanh", "10. Xác suất ", "11. Chơi ca-rô", "12. Tạo mật khẩu", "13. Banking", "14. Nhận xét", "15. Đăng xuất"],

   } 
   for item in menus[language]:
     print(item)
   return [(input("Chọn 1 lựa chọn: "))]

def download_audio_from_youtube(youtube_url, filename="song"):
    try:
        # Create a YouTube object
        yt = YouTube(youtube_url)
        
        # Get the highest quality audio stream available
        audio_stream = yt.streams.filter(only_audio=True).first()

        # Download the audio file
        audio_file = audio_stream.download(filename=filename)

        # Convert the downloaded file to mp3 using pydub
        audio = AudioSegment.from_file(audio_file)
        mp3_filename = f"{filename}.mp3"
        audio.export(mp3_filename, format="mp3")

        # Remove the original file to clean up
        os.remove(audio_file)

        return mp3_filename
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def play_song(filename):
    try:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Load the song
        pygame.mixer.music.load(filename)

        # Play the song
        pygame.mixer.music.play()

        # Wait for the song to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()
        pygame.mixer.quit()   

    except Exception as e:
        print(f"An error occurred: {e}")

def binary_convert():
  try:  
    ans = int(input("Nhập số: "))
    x = bin(ans)
    return print("Bằng: ", x)
  except Exception:
    print("Lỗi! Vui lòng nhập 1 con số hợp lệ") 


def create_ticket():
    # Tạo một mật khẩu ngẫu nhiên dài 8 ký tự
    characters = string.ascii_letters + string.digits
    ticket = ''.join(random.choice(characters) for i in range(8))
    # Lưu mật khẩu vào file pwd.txt
    with open("ticket.txt", 'w') as file:
        file.write(ticket)
    return ticket

def send_feedback_via_email(feedback, feedback_email, recipient_email, stars):
    # Cấu hình thông tin tài khoản Gmail của bạn
    sender_email = "formyapplicationdonotreply@gmail.com" # Thay bằng email của bạn
    sender_password = "nkmp dojx nnbj anox" # Thay bằng mật khẩu ứng dụng
    

    # Cấu hình email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"Phản hồi từ người dùng {feedback_email}"
    body = f"""Chào {recipient_email}! \n"""
    body_2 = f"""Phản hồi của người dùng {feedback_email} là: {feedback}. \n"""
    body_6 = f"""Đánh giá phần mềm: {stars} sao. \n"""
    body_7 = """Hệ thống sẽ lưu lại phản hồi này dưới dạng email. \n"""
    body_3 = """Bạn không được phép trả lời mail này. \n""" 
    body_4 = """Xin cảm ơn, \n"""
    body_5 = """The Administrator Team."""
    msg.attach(MIMEText(body, 'plain'))
    msg.attach(MIMEText(body_2, 'plain'))
    msg.attach(MIMEText(body_6, 'plain'))
    msg.attach(MIMEText(body_7, 'plain'))
    msg.attach(MIMEText(body_3, 'plain'))
    msg.attach(MIMEText(body_4, 'plain'))
    msg.attach(MIMEText(body_5, 'plain'))
    
    try:
        # Thiết lập kết nối và gửi email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
    except Exception as e:
        print(f"Không thể gửi email: {e}") 

def feedback():
   while True: 
    a = input("Nhận xét của người dùng: ")
    e = input("Bạn đánh giá phần mềm này mấy sao trên thang điểm 1-5: ")
    d = input("Bạn có muốn thoát không: ")
    if 'yes' in d:
       break 
    
    ticket_code = create_ticket()
    print("------------------------------------------------------")
    print("Mã vé của bạn là:  " + ticket_code)
    print("Bạn có 15 phút để xác nhận rằng bạn không phải là bot.")
    print("**Chú ý dấu cách, space.**")
    print("Xin cảm ơn,")
    print("The Administrator Team.")
    print("------------------------------------------------------")
    b = input("Mã vé: ")
        
    with open("ticket.txt", 'r') as file:
        stored_code = file.read().strip()
        if b != stored_code:
            print("Sai!")
            break
        else:
            print("Đúng!")
    
    c = input("Xác nhận?(Y/N): ")
    if c.lower() == 'y':
        feedback_email = input("Email của bạn là gì?: ")
        recipient_email = "nguyendinhhung240112@gmail.com" 
        send_feedback_via_email(a, feedback_email, recipient_email, e)
    else:
        print("Phản hồi không được gửi.")
    print("Đang gửi phản hồi...")
    time.sleep(10)
    print("Phản hồi đã được gửi qua email thành công!")

    
def convert_unit():
  try:  
    ans = int(input("Nhập số mét: "))
    
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

def quick_response_code():
   img = qrcode.make('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
   type(img)  
   img.save("some_file.png")

def one_two_three():
  
  choices = ['kéo', 'búa', 'bao']
  
# Hàm xác định kết quả của trò chơi
  def determine_winner(player_choice, computer_choice):
    if player_choice == computer_choice:
        return 'hòa'
    elif (player_choice == 'kéo' and computer_choice == 'bao') or \
         (player_choice == 'búa' and computer_choice == 'kéo') or \
         (player_choice == 'bao' and computer_choice == 'búa'):
        return 'thắng'
    else:
        return 'thua'

# Số lần mô phỏng
  n_simulations = 10000

# Khởi tạo biến đếm cho các kết quả
  results = {'thắng': 0, 'thua': 0, 'hòa': 0}

# Mô phỏng trò chơi
  for _ in range(n_simulations):
    player_choice = random.choice(choices)
    computer_choice = random.choice(choices)
    result = determine_winner(player_choice, computer_choice)
    results[result] += 1

# Tính xác suất
  win_prob = results['thắng'] / n_simulations * 100
  lose_prob = results['thua'] / n_simulations * 100
  draw_prob = results['hòa'] / n_simulations * 100

  print(f'Xác suất thắng: {win_prob:.4f}')
  print(f'Xác suất thua: {lose_prob:.4f}')
  print(f'Xác suất hòa: {draw_prob:.4f}') 
# Tính xác xuất khi tung xúc xắc
def Dice():
  n_simulations = 1000
  results = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
  choices = ['1', '2', '3', '4', '5', '6']

  def tính_xác_xuất():
    one_prob = results['1'] / n_simulations * 100
    two_prob = results['2'] / n_simulations * 100
    three_prob = results['3'] / n_simulations * 100
    four_prob = results['4'] / n_simulations * 100
    five_prob = results['5'] / n_simulations * 100
    six_prob = results['6'] / n_simulations * 100
    print(f'Xác suất ra mặt 1 nút là: {one_prob:.4f}')
    print(f'Xác suất ra mặt 2 nút là: {two_prob:.4f}')
    print(f'Xác suất ra mặt 3 nút là: {three_prob:.4f}')
    print(f'Xác suất ra mặt 4 nút là: {four_prob:.4f}')
    print(f'Xác suất ra mặt 5 nút là: {five_prob:.4f}')
    print(f'Xác suất ra mặt 6 nút là: {six_prob:.4f}')
    return one_prob, two_prob, three_prob, four_prob, five_prob, six_prob

  for _ in range(n_simulations):
    result = random.choice(choices)
    results[result] += 1

  tính_xác_xuất()

def print_board(board):
    for row in board:
        print(" ".join([str(cell) if cell is not None else '.' for cell in row]))
    print()

def check_win(board, player):
    for row in range(5):
        if all([cell == player for cell in board[row]]):
            return True
    for col in range(5):
        if all([board[row][col] == player for row in range(5)]):
            return True
    if all([board[i][i] == player for i in range(5)]) or all([board[i][4-i] == player for i in range(5)]):
        return True
    return False

def check_draw(board):
    return all([all([cell is not None for cell in row]) for row in board])

def minimax(board, depth, alpha, beta, is_maximizing, max_depth):
    if check_win(board, 'O'):
        return 1
    if check_win(board, 'X'):
        return -1
    if check_draw(board):
        return 0
    if depth >= max_depth:
       return 0

    if is_maximizing:
        best_score = -math.inf
        for row in range(5):
            for col in range(5):
                if board[row][col] is None:
                    board[row][col] = 'O'
                    score = minimax(board, depth + 1, alpha, beta, False, max_depth)
                    board[row][col] = None
                    best_score = max(score, best_score)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
        return best_score
    else:
        best_score = math.inf
        for row in range(5):
            for col in range(5):
                if board[row][col] is None:
                    board[row][col] = 'X'
                    score = minimax(board, depth + 1, alpha, beta, True, max_depth)
                    board[row][col] = None
                    best_score = min(score, best_score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
        return best_score

def best_move(board, max_depth):
    best_score = -math.inf
    move = None
    for row in range(5):
        for col in range(5):
            if board[row][col] is None:
                board[row][col] = 'O'
                score = minimax(board, 0, -math.inf, math.inf, False, max_depth)
                board[row][col] = None
                if score > best_score:
                    best_score = score
                    move = (row, col)
    return move


# Khởi tạo bảng cờ ca rô 3x3
board = [
    [None, None, None, None, None],
    [None, None, None, None, None],
    [None, None, None, None, None],
    [None, None, None, None, None],
    [None, None, None, None, None],
]

def banking():
   def withdraw():
      a = int(input("Nhập số tiền bạn muốn gửi vào tài khoản ngân hàng của bạn: "))
      if a < 0:
         print("Vui lòng nhập 1 con số hợp lệ")
         return 0
      else:
         return a
   def view(balance):
     print(f"Số tiền của bạn là ${balance:.2f}")

   def info():
      pass   
      
   def deposit(balance):
     a = int(input("Nhập số tiền bạn muốn rút ra: ")) 
     if a > balance:
        print("Vui lòng nhập 1 con số hợp lệ")
        return 0
     elif a < 0:
        print("Vui lòng nhập 1 con số hợp lệ")
        return 0 
     else:
        return a
   def main(): 
    balance = 0     
    is_running = True
    while is_running:
     print("-------------------------")
     print("Hệ thống ngân hàng")
     print("1. Xem số dư tài khoản ngân hàng.")
     print("2. Rút tiền")
     print("3. Chuyển tiền vào tài khoản ngân hàng.")
     print("4. Exit.")
     ans = input("Chọn 1 lựa chọn: ")
     if ans == '1':
       view(balance)
     elif ans == '2':
       balance -= deposit(balance)
     elif ans == '3':
       balance += withdraw()
     elif ans == '4':
       print("Cảm ơn vì đã sử dụng hệ thống ngân hàng")   
       is_running = False
   if __name__ == '__main__':
     main()          

def create_pwd(pw_length=8):
   letters = string.ascii_letters
   digits = string.digits
   special_chars = string.punctuation

   alphabets = letters + digits + special_chars
   pwd = ''
   pwd_strong =  False

   while not pwd_strong:
      pwd += ''.join(secrets.choice(alphabets))
      if (any(char in special_chars for char in pwd) and sum(char in digits for char in pwd) >= pw_length):
        pwd_strong = True
      with open("pwd.txt", 'w') as files:
         files.write(pwd) 
                           
   return pwd

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
 except ZeroDivisionError:
    print("Không thể chia cho 0!")

def magnetic():
   youtube_url = "https://www.youtube.com/watch?v=Vk5-c_v4gMU"
   audio_file = download_audio_from_youtube(youtube_url)
   if audio_file:
        play_song(audio_file)
        
       # Optionally, remove the song file after playing
        os.remove(audio_file)
   else:
        print("Failed to download audio.")

def co_duyen_khong_no():
   youtube_url = "https://www.youtube.com/watch?v=BlFcXAYMr1M"
   audio_file = download_audio_from_youtube(youtube_url)
   if audio_file:
        play_song(audio_file)
        
       # Optionally, remove the song file after playing
        os.remove(audio_file)
   else:
        print("Failed to download audio.")

def the_fat_rat_1():
   youtube_url = "https://www.youtube.com/watch?v=cMg8KaMdDYo"
   audio_file = download_audio_from_youtube(youtube_url)
   if audio_file:
        play_song(audio_file)
        
       # Optionally, remove the song file after playing
        os.remove(audio_file)
   else:
        print("Failed to download audio.") 

def the_fat_rat_2():
   youtube_url = "https://www.youtube.com/watch?v=B7xai5u_tnk"
   audio_file = download_audio_from_youtube(youtube_url)
   if audio_file:
        play_song(audio_file)
        
       # Optionally, remove the song file after playing
        os.remove(audio_file)
   else:
        print("Failed to download audio.")

def victory():
  youtube_url = "https://www.youtube.com/watch?v=hKRUPYrAQoE"
  audio_file = download_audio_from_youtube(youtube_url)
  if audio_file:
        play_song(audio_file)
        
       # Optionally, remove the song file after playing
        os.remove(audio_file)
  else:
        print("Failed to download audio.")                         
     
async def main():
 try:
  login_and_register()
  while True:
    print(f"started at {time.strftime('%X')}")
    language = 'en'
    a, b, c, menu, goodbye = input_user(language)
    a,b = int(float(a)), int(float(b))
    if 'yes' in goodbye:
     break
    if 'yes' in menu:
     opt = men(language)
     if '3' in opt:
      x.add_column("Column 1", )
      x.add_column("Column 2", )
      x.add_column("Column 3", )
      print(x)
     elif '13' in opt:
        banking() 
     elif '12' in opt:
        print(create_pwd()) 
     elif '14' in opt:
        feedback()
     elif '15' in opt:
        login_and_register()      
     elif '11' in opt:
      print("Bắt đầu trò chơi cờ ca rô!")
      print_board(board)

      while True:
    # Player X move
       row = int(input("Nhập hàng (0, 1, 2, 3, 4): "))
       col = int(input("Nhập cột (0, 1, 2, 3, 4): "))
       if board[row][col] is None:
        board[row][col] = 'X'
       else:
        print("Ô đã được chọn. Vui lòng chọn ô khác.")
        continue
    
       print_board(board)
       if check_win(board, 'X'):
        print("Người chơi X thắng!")
        break
       if check_draw(board):
        print("Hòa!")
        break

    # Bot O move
       print("Bot O đang suy nghĩ...")
       max_depth = 3
       move = best_move(board, max_depth)
       if move is not None:
        board[move[0]][move[1]] = 'O'
        print("Bot O đi: hàng {}, cột {}".format(move[0], move[1]))
        print_board(board)
        if check_win(board, 'O'):
            print("Bot O thắng!")
            break
        if check_draw(board):
            print("Hòa!")
            break
       else:
        print("Hòa!")
        break
     elif '10' in opt:
       print("1. Tung xúc xắc")
       print("2. Kéo búa bao")
       ans = input("Chọn 1 lựa chọn: ")
       if '2' in ans:
         one_two_three()
       if '1' in ans:
          Dice()  
     elif '4' in opt:
       binary_convert()
     elif '8' in opt:
       print("Trong thư viện nhạc có: ")
       print("1. The Fat Rat 1")
       print("2. The Fat Rat 2")
       print("3. Victory")
       print("4. Có duyên không nợ")
       print("5. Magnetic")
       ans = input("Chọn 1 lựa chọn: ")
       if '1' in ans:
         the_fat_rat_1()
       if '2' in ans:
          the_fat_rat_2()
       if '3' in ans:
          victory()
       if '4' in ans:
          co_duyen_khong_no()
       if '5' in ans:
          magnetic()            
     elif '9' in opt:
        quick_response_code()
        ans = input("Bạn có muốn xóa mã qr không?")
        if 'yes' in ans:
         os.remove("some_file.png")    
     elif '5' in opt:
      print("1. Độ dài")
      print("2. Khối lượng")
      ans = input("Chọn 1 lựa chọn: ")
      if '1' in ans:
        convert_unit()
      if '2' in ans:
        convert_unit_pound()
     elif '6' in opt:
      print("1. Kiểm tra số nguyên tố")
      print("2. Tìm số nguyên tố")
      ans = input("Chọn 1 lựa chọn: ")
      if '1' in ans:
        prime_check()     
      if '2' in ans:
       eratosthenes()
     elif '7' in opt:
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
    equal = calculate(a,b,c)
    print(f"Bằng:  {equal}")
    print(f"finished at {time.strftime('%X')}")
 except ValueError:
     print("Lỗi! Vui lòng nhập 1 con số hợp lệ")
   
if __name__ == "__main__" :
    asyncio.run(main()) 
   
