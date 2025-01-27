import os
import platform

import colorama
from colorama import Fore, Back
import pandas as pd



# ======================================================
# Initialize colorama
colorama.init() 

# ======================================================
# function for os agnostic clear screen
def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")   

# ======================================================
def print_header(text:str = "", fill_char: str = " ", color=Fore.WHITE,bg_color=Back.BLUE, new_line_padding: int = 1):

    # line padding
    line_padding(new_line_padding)

    # get valid color
    color = get_valid_color(color)

    # get valid bg_color
    bg_color = get_valid_bg_color(bg_color)

    # prepare text
    text = prepare_text(text)

    # if fill_char len > 1, set it to "-", allowes for blank 
    if len(fill_char) > 1:
        fill_char = "-"

    # create header
    header_text = fill_char * 3 + text

    # get screen width
    screen_width = os.get_terminal_size().columns

    # get right padding
    right_padding = (screen_width - len(header_text))

    # append right padding to header
    header_text = header_text + fill_char * right_padding

    # print the header
    print(f"{color}{bg_color}{header_text}{Fore.RESET}{Back.RESET}")

# ======================================================
def prepare_text(text:str):
    
    # default text to empty string if None
    if text is None:
        return ""
    
    # default text to empty string if empty string
    if len(text) <= 0:
        return ""
    
    # remove leading and trailing whitespace
    text = text.strip()
    
    # add a space before and after the text
    text = " " + text + " "

    return text
# ======================================================
def line_padding(new_line_padding: int = 0):
    if new_line_padding <= 0:
        return

    # for loop from 0 to new_line_padding
    for i in range(new_line_padding):
        print(f"{Fore.RESET}{Back.RESET}")

# ======================================================
# function to print to screen
def print_text(text: str, color=Fore.WHITE, new_line_padding: int = 0):

    # line padding
    line_padding(new_line_padding)

    # get valid color
    color = get_valid_color(color)

    # Print the text with the specified foreground color
    print(f"{color}{text}{Fore.RESET}{Back.RESET}")

# ======================================================
def get_valid_color(color: str):
    if color not in vars(Fore).values():
        return Fore.RESET
    return color

# ======================================================
def get_valid_bg_color(bg_color: str):
    if bg_color not in vars(Back).values():
        return Back.RESET
    return bg_color

# ======================================================
# function to print dataframe
def data_frame(df: pd.DataFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


# ######################################################
# ######################################################
# if main statement
if __name__ == "__main__":
    clear_screen()
    print("\n--- header() tests ---")
    print_header()
    print_header("header", "asdfsd")
    print_header("header", "*")
    print_header("header", "*", Fore.RED, Back.YELLOW)    
    print_header("line padding: 1", "*", Fore.RED, Back.BLUE, 1)    
    print_header("line padding: 2", "*", Fore.RED, Back.GREEN, 2)    
    print_header(fill_char=" ", bg_color=Back.MAGENTA, new_line_padding=1) 
    print_header("END header() tests", fill_char=" ", bg_color=Back.RED, new_line_padding=2) 

    print_header("text() tests", bg_color=Back.BLUE, new_line_padding=1)
    print_text("")
    print_text("text")
    print_text("text", Fore.GREEN)
    print_text("text", Fore.GREEN, 1)
    print_text("text", Fore.GREEN, 2)
    print_text(f"{Fore.YELLOW}fString")
    print_header("END text() tests", fill_char=" ", bg_color=Back.RED, new_line_padding=2) 

    # ------------------------------------------------------
    print_header("data_frame() tests", bg_color=Back.BLUE, new_line_padding=1)
    
    
    data = {
        "Name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hannah", "Ivy", "Jack"],
        "Age": [25, 30, 35, 28, 22, 33, 29, 31, 24, 26],
        "Score": [85, 90, 95, 88, 76, 92, 89, 91, 84, 87],
        "City": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"],
        "Department": ["HR", "Engineering", "Sales", "Marketing", "Finance", "IT", "Support", "Logistics", "R&D", "Administration"],
        "Salary": [65000, 85000, 75000, 70000, 62000, 88000, 69000, 72000, 64000, 71000],
        "Years_Exp": [3, 5, 7, 4, 2, 6, 4, 5, 3, 4],
        "Projects": [4, 6, 8, 5, 3, 7, 5, 6, 4, 5],
        "Performance": [4.2, 4.5, 4.8, 4.3, 3.9, 4.6, 4.4, 4.5, 4.1, 4.3],
        "Education": ["Bachelors", "Masters", "PhD", "Bachelors", "Bachelors", "Masters", "Bachelors", "Masters", "Bachelors", "Masters"],
        "Remote_Days": [2, 3, 1, 4, 5, 2, 3, 2, 4, 3],
        "Team_Size": [8, 12, 15, 10, 6, 14, 9, 11, 7, 10],
        "Training_Hours": [20, 35, 40, 25, 15, 38, 22, 30, 18, 28],
        "Certifications": [2, 4, 5, 3, 1, 4, 2, 3, 2, 3],
        "Leave_Days": [15, 18, 20, 16, 14, 19, 16, 17, 15, 17],
        "Client_Rating": [4.3, 4.6, 4.9, 4.4, 4.0, 4.7, 4.5, 4.6, 4.2, 4.4],
        "Languages": [3, 4, 5, 3, 2, 4, 3, 4, 2, 3],
        "Overtime_Hours": [10, 15, 12, 8, 5, 14, 9, 11, 7, 10],
        "Travel_Days": [5, 8, 10, 6, 3, 9, 6, 7, 4, 7],
        "Bonus": [2000, 3500, 4000, 2500, 1500, 3800, 2200, 3000, 1800, 2800]
       
    }
    


    # create dataframe
    df = pd.DataFrame(data)


    # Call the function
    data_frame(df)



# ######################################################
# ######################################################