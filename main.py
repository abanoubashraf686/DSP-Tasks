import tkinter as tk
import os
import subprocess

list_text = ["Task 1&2","Task 3","Task 4","Task 5 to 9"]

def run_script(button_number):
    script_name = f"{list_text[button_number]}.py"
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(script_path)
    if script_path:
        subprocess.run(["python", script_path],shell=True)
    else:
        print(f"Script '{script_name}' not found.")

# Create the main window
root = tk.Tk()
root.title("Script Runner")

# Function to create a button with a specific number
def create_button(button_number):
    return tk.Button(root, text=list_text[button_number], command=lambda: run_script(button_number))

# Create and grid the buttons
for i in range(4):
    button = create_button(i)
    button.grid(row=i, column=1, padx=60)

# Start the main event loop
root.mainloop()
