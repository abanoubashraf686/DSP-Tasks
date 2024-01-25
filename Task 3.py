import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import math

# Function to update entry value from slider
def update_entry_value(slider, entry_var):
    entry_var.set(slider.get())

def generate_signal_discrete(n, levels, quantized_signal):
    print(os.path.join(os.getcwd(), 'QuanOutput.txt'))
    with open('QuanOutput.txt', 'w') as file:
        file.write("0\n0\n")
        file.write(f'{n}\n')
        for i in range(n):
            file.write(f'{str(levels[i])} {quantized_signal[i]}\n')

# Function to plot the selected function
def plot_function():
    function_type = function_choice.get()
    list = []
    for i in range(6):
        list.append(arr_of_sliders[i].get())
    [A, theta, F, Fs, number_of_cycles, num_levels] = list
    # Check the value of units_choice
    selected_units = units_choice.get()

    # Now you can use the selected_units variable to check the chosen units
    if selected_units == "Levels":
        # The "Levels" option is selected
       num_levels = int(num_levels)
    elif selected_units == "Bits":
        # The "Bits" option is selected
        num_levels = pow(2, int(num_levels))
    if Fs < 2 * F:
        messagebox.showerror("Sampling Theorem Violation", "The sampling frequency (Fs) must be at least twice the analog frequency (F).")
        return

    t_continuous = np.linspace(0, number_of_cycles / F, int(number_of_cycles * Fs / F))
    n = np.arange(0, int(number_of_cycles * Fs / F))
    signal_continuous = 0
    signal_discrete = 0
    quantized_signal = 0
    quantization_error = 0

    if function_type == 'Sine':
        signal_continuous = A * np.sin(2 * np.pi * F * t_continuous + theta)
        signal_discrete = A * np.sin(2 * np.pi * F / Fs * n + theta)
    elif function_type == 'Cosine':
        signal_continuous = A * np.cos(2 * np.pi * F * t_continuous + theta)
        signal_discrete = A * np.cos(2 * np.pi * F / Fs * n + theta)
    elif function_type == 'Both':
        signal_continuous_sine = A * np.sin(2 * np.pi * F * t_continuous + theta)
        signal_continuous_cosine = A * np.cos(2 * np.pi * F * t_continuous + theta)

        signal_discrete_sine = A * np.sin(2 * np.pi * F / Fs * n + theta)
        signal_discrete_cosine = A * np.cos(2 * np.pi * F / Fs * n + theta)

        # Plot sin in blue and cos in orange
        ax1.clear()
        ax1.plot(t_continuous, signal_continuous_sine, color='blue', label='Sin')
        ax1.plot(t_continuous, signal_continuous_cosine, color='orange', label='Cos')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Continuous Signal')
        ax1.legend()

        ax2.clear()
        ax2.stem(n, signal_discrete_sine, linefmt='b-', markerfmt='bo', basefmt=' ', label='Sin')
        ax2.stem(n, signal_discrete_cosine, linefmt='orange', markerfmt='o', basefmt=' ', label='Cos')
        ax2.set_xlabel('Sample Index (n)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Discrete Signal')
        ax2.legend()

        # Calculate quantized signal and quantization error
        quantized_signal_sine = np.round(num_levels * (signal_discrete_sine - min(signal_discrete_sine)) / (
                max(signal_discrete_sine) - min(signal_discrete_sine)))
        quantized_signal_cosine = np.round(num_levels * (signal_discrete_cosine - min(signal_discrete_cosine)) / (
                max(signal_discrete_cosine) - min(signal_discrete_cosine)))

        quantization_error_sine = signal_discrete_sine - (
                min(signal_discrete_sine) + quantized_signal_sine * (
                max(signal_discrete_sine) - min(signal_discrete_sine)) / num_levels)
        quantization_error_cosine = signal_discrete_cosine - (
                min(signal_discrete_cosine) + quantized_signal_cosine * (
                max(signal_discrete_cosine) - min(signal_discrete_cosine)) / num_levels)

        ax3.clear()
        ax3.plot(n, quantized_signal_sine, linefmt='b-', markerfmt='bo', basefmt=' ',
                 label='Quantized Sin')
        ax3.plot(n, quantized_signal_cosine, linefmt='orange', markerfmt='o', basefmt=' ',
                 label='Quantized Cos')
        ax3.set_xlabel('Sample Index (n)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Quantized Signal')
        ax3.legend()

        ax4.clear()
        ax4.stem(n, quantization_error_sine, linefmt='b-', markerfmt='bo', basefmt=' ',
                 label='Quantization Error Sin')
        ax4.stem(n, quantization_error_cosine, linefmt='orange', markerfmt='o', basefmt=' ',
                 label='Quantization Error Cos')
        ax4.set_xlabel('Sample Index (n)')
        ax4.set_ylabel('Amplitude')
        ax4.set_title('Quantization Error')
        ax4.legend()

        canvas.draw()
        # generate_signal_discrete(list, True)
        # generate_signal_discrete(list, False)
        # generate_signal_continuous(list)
        return

    ax1.clear()
    ax1.plot(t_continuous, signal_continuous)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Continuous Signal')

    ax2.clear()
    ax2.stem(n, signal_discrete)
    ax2.set_xlabel('Sample Index (n)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Discrete Signal')

    # Calculate quantized signal and quantization error
    quantized_signal = np.round(num_levels * (signal_discrete - min(signal_discrete)) / (
            max(signal_discrete) - min(signal_discrete)))
    quantization_error = signal_discrete - (
            min(signal_discrete) + quantized_signal * (max(signal_discrete) - min(signal_discrete)) / num_levels)



    ax3.clear()
    ax3.plot(n, quantized_signal, label='Quantized')
    ax3.set_xlabel('Sample Index (n)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Quantized Signal')

    ax4.clear()
    ax4.stem(n, quantization_error, linefmt='b-', markerfmt='bo', basefmt=' ',
             label='Quantization Error')
    ax4.set_xlabel('Sample Index (n)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Quantization Error')

    canvas.draw()

APE = 0
# Function to plot the loaded signal
def plot_loaded_signal():
    global APE
    file_path = file_path_var.get()
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            signal_type = int(lines[0].strip())
            is_periodic = int(lines[1].strip())
            n1 = int(lines[2].strip())
            data = lines[3:]
            if signal_type == 0:  # Time
                time = []
                amplitude = []
                for line in data:
                    parts = list(line.strip().split(' '))
                    time.append(parts[0])
                    amplitude.append(float(parts[1]))

                ax1.clear()
                X = np.array(time)
                Y = np.array(amplitude)
                ax1.plot(X, Y)
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Amplitude')
                ax1.set_title('Continuous Signal')

                ax2.clear()
                ax2.stem(time, amplitude)
                ax2.set_xlabel('Sample Index (n)')
                ax2.set_ylabel('Amplitude')
                ax2.set_title('Discrete Signal')

                # Calculate quantized signal and quantization error
                num_levels = int(arr_of_sliders[5].get())
                # Check the value of units_choice
                selected_units = units_choice.get()

                # Now you can use the selected_units variable to check the chosen units
                if selected_units == "Levels":
                    # The "Levels" option is selected
                    num_levels = int(num_levels)
                elif selected_units == "Bits":
                    # The "Bits" option is selected
                    num_levels = pow(2, int(num_levels))
                mn = min(amplitude)
                mx = max(amplitude)
                print(amplitude)
                delta = (mx - mn) / num_levels
                levels = []
                quantized_signal = []
                quantization_error = []
                EPE = []
                ranges = []
                start = mn
                for index in range(num_levels):
                    ranges.append((round(start,7),round(start+delta,7)))
                    start+=delta
                sz = len(ranges)
                for amp in amplitude:
                    idx =0
                    for (mini,maxi) in ranges:
                        mid = mini+delta/2
                        if mini <= amp <= maxi:
                            levels.append(idx)
                            EPE.append(round(mid-amp,3))
                            quantized_signal.append(round(mid,5))
                            quantization_error.append((mid - amp) * (mid - amp))
                            break
                        idx+=1

                APE = sum(quantization_error) / len(amplitude)
                ape_var.set(f"APE: {APE}")
                BinaryLevels = decimals_to_binary_with_x_bits(levels,math.ceil(math.log2(num_levels)))
                generate_signal_discrete(len(BinaryLevels), BinaryLevels, quantized_signal)
                ax3.clear()
                ax3.plot(time, quantized_signal)
                ax3.set_xlabel('Sample Index (n)')
                ax3.set_ylabel('Amplitude')
                ax3.set_title('Quantized Signal')

                ax4.clear()
                ax4.stem(time, quantization_error)
                ax4.set_xlabel('Sample Index (n)')
                ax4.set_ylabel('Amplitude')
                ax4.set_title('Quantization Error')
                file_name = os.path.basename(file_path)
                if file_name=="Quan1_input.txt":
                    QuantizationTest1("Quan1_Out.txt", BinaryLevels, quantized_signal)
                elif file_name =="Quan2_input.txt":
                    levels2=[]
                    for x in levels:
                        x+=1
                        levels2.append(x)
                    print(levels2,BinaryLevels,quantized_signal,EPE)
                    QuantizationTest2("Quan2_Out.txt",levels2,BinaryLevels, quantized_signal, EPE)
                canvas.draw()
            else:
                print("HERE 351")
                messagebox.showerror("Invalid Signal Type", "Invalid signal type in the file.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def decimals_to_binary_with_x_bits(decimal_list, num_bits):
    binary_list = []
    for decimal in decimal_list:
        # Check if the number of bits is sufficient to represent the decimal number
        if decimal < 0 or decimal >= 2**num_bits:
            binary_list.append("Invalid input")
        else:
            binary_representation = bin(decimal)[2:]  # Convert to binary, remove '0b' prefix
            if len(binary_representation) < num_bits:
                # If the binary representation is shorter than x bits, pad it with zeros
                binary_representation = '0' * (num_bits - len(binary_representation)) + binary_representation
            else:
                # If it's longer than x bits, truncate it to the first x bits
                binary_representation = binary_representation[:num_bits]
            print(decimal,binary_representation)
            binary_list.append(binary_representation)
    return binary_list


# Function to browse and load file
def browse_file():
    file_path = filedialog.askopenfilename()
    file_path_var.set(file_path)









app = tk.Tk()
app.title("Function Plotter")

# Set the default window size to 1080p resolution
app_width = 1920
app_height = 1080
app.geometry(f"{app_width}x{app_height}")

# Variable for function choice
function_choice = tk.StringVar()
function_choice.set('Sine')

# Variables for input fields
arr_of_var = []
for i in range(6):
    x = tk.StringVar()
    arr_of_var.append(x)

frame = ttk.Frame(app)
frame.pack(expand=True, fill='both', padx=10, pady=10)

# Create a big title label at the top
big_title_label = ttk.Label(frame, text="Function Plotter", font=("Arial", 24))
big_title_label.grid(row=0, column=0, columnspan=3, padx=5, pady=20, sticky='n')

# Create a frame to center the elements in the middle
center_frame = ttk.Frame(frame)
center_frame.grid(row=1, column=0, columnspan=3)

# Entry and Browse button for file path
file_path_var = tk.StringVar()
file_path_entry = ttk.Entry(center_frame, textvariable=file_path_var, width=40)
file_path_entry.grid(row=8, column=0, padx=5, pady=5, columnspan=2, sticky='w')
browse_button = ttk.Button(center_frame, text="Browse", command=browse_file)
browse_button.grid(row=8, column=1, padx=5, pady=5, sticky='e')

# Load button for loaded signal
load_button = ttk.Button(center_frame, text="Load Signal", command=plot_loaded_signal)
load_button.grid(row=8, column=2, columnspan=1, padx=5, pady=10)

# Label and Combobox for function choice
function_label = ttk.Label(center_frame, text="Select Function:")
function_label.grid(row=0, column=0, padx=5, pady=5)
function_combobox = ttk.Combobox(center_frame, textvariable=function_choice, values=['Sine', 'Cosine', 'Both'], state='readonly')
function_combobox.grid(row=0, column=1, padx=5, pady=5)

arr_of_text = ["Amplitude (A):", "Phase Shift (θ) (radians):", "Analog Frequency (F) (Hz):",
               "Sampling Frequency (Fs):", "Number of cycles in the plot:", "Number of Levels (or Bits):"]
arr_of_sliders = []

for i in range(6):
    label = ttk.Label(center_frame, text=arr_of_text[i])
    label.grid(row=i + 1, column=0, padx=5, pady=5)
    slider = ttk.Scale(center_frame, from_=0, to=10, orient='horizontal', length=200, variable=arr_of_var[i])
    slider.grid(row=i + 1, column=1, padx=5, pady=5)
    entry = ttk.Entry(center_frame, textvariable=arr_of_var[i])
    entry.grid(row=i + 1, column=2, padx=5, pady=5)
    arr_of_var[i].set(slider.get())
    arr_of_sliders.append(slider)


# Create radio buttons for Levels and Bits
levels_bits_frame = ttk.Frame(center_frame)
levels_bits_frame.grid(row=6, column=3, columnspan=2, padx=5, pady=5, sticky='w')

levels_bits_label = ttk.Label(levels_bits_frame, text="Choose units:")
levels_bits_label.grid(row=0, column=0, padx=5, pady=5)

units_choice = tk.StringVar()
units_choice.set('Levels')

levels_radio = ttk.Radiobutton(levels_bits_frame, text="Levels", variable=units_choice, value="Levels")
levels_radio.grid(row=0, column=1, padx=5, pady=5, sticky='w')

bits_radio = ttk.Radiobutton(levels_bits_frame, text="Bits", variable=units_choice, value="Bits")
bits_radio.grid(row=0, column=2, padx=5, pady=5, sticky='e')

# Plot button
plot_button = ttk.Button(center_frame, text="Plot Signal and generate output files", command=plot_function)
plot_button.grid(row=7, columnspan=3, padx=5, pady=10)



# Create a tkinter StringVar for APE and initialize it
ape_var = tk.StringVar()
ape_var.set("APE: N/A")

# Add a Label widget to display APE
ape_label = ttk.Label(center_frame, textvariable=ape_var)
ape_label.grid(row=9, columnspan=3, padx=5, pady=10)
ape_var.set(f"APE: {APE}")

# Create Matplotlib figures and canvas
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(app_width / 96, app_height / 192), dpi=96)
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

# Bind events to update entry fields
for i in range(6):
    arr_of_sliders[i].bind("<Motion>", lambda event: update_entry_value(arr_of_sliders[i], arr_of_var[i]))

app.mainloop()