import cmath
import math
import os
import shutil
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.constants import *

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def generate_file(name, is_freq, is_periodic, x, y):
    try:
        n = len(x)
        with open(f"{name}.txt", 'w') as file:
            file.write(f'{is_freq}\n{is_periodic}\n{n}\n')
            for i in range(n):
                file.write(f'{x[i]} {y[i]}\n')
    except Exception as e:
        messagebox.showerror("Error", f": {str(e)}")


def Compare_Signals(file_name, Your_indices, Your_samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            elif len(L.split('  ')) == 2:
                L = line.split('  ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print(Fore.GREEN + "Test case passed successfully" + Style.RESET_ALL)


InputSignal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
               19.0, 20.0,
               21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
               38.0, 39.0, 40.0,
               41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
               58.0, 59.0, 60.0,
               61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0,
               78.0, 79.0, 80.0,
               81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0,
               98.0, 99.0, 100.0]


def DerivativeSignal(FirstDrev, SecondDrev):
    expectedOutput_first = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1]
    expectedOutput_second = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0]

    """
    Write your Code here:
    Start
    """
    """
    End
    """
    """
    Testing your Code
    """
    if ((len(FirstDrev) != len(expectedOutput_first)) or (len(SecondDrev) != len(expectedOutput_second))):
        print("mismatch in length")
        return
    first = second = True
    for i in range(len(expectedOutput_first)):
        if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
            continue
        else:
            first = False
            print("1st derivative wrong")
            return
    for i in range(len(expectedOutput_second)):
        if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
            continue
        else:
            second = False
            print("2nd derivative wrong")
            return
    if (first and second):
        print("Derivative Test case passed successfully")
    else:
        print("Derivative Test case failed")
    return


def Shift_Fold_Signal(file_name, Your_indices, Your_samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Shift_Fold_Signal Test case failed, your signal have different values from the expected one")
            return
    print("Shift_Fold_Signal Test case passed successfully")


def SignalSamplesAreEqual(file_name, samples):
    """
    this function takes two inputs the file that has the expected results and your results.
    file_name : this parameter corresponds to the file path that has the expected output
    samples: this parameter corresponds to your results
    return: this function returns Test case passed successfully if your results is similar to the expected output.
    """
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    if len(expected_samples) != len(samples):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print("Test case passed successfully")


def ConvTest(Your_indices, Your_samples):
    """
    Test inputs
    InputIndicesSignal1 =[-2, -1, 0, 1]
    InputSamplesSignal1 = [1, 2, 1, 1 ]

    InputIndicesSignal2=[0, 1, 2, 3, 4, 5 ]
    InputSamplesSignal2 = [ 1, -1, 0, 0, 1, 1 ]
    """

    expected_indices = [-2, -1, 0, 1, 2, 3, 4, 5, 6]
    expected_samples = [1.0, 1.0, -1.0, 0.0, 0.0, 3.0, 3.0, 2.0, 1.0]

    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Conv Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Conv Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Conv Test case failed, your signal have different values from the expected one")
            return
    print("Conv Test case passed successfully")


def DFT(samples_list):
    amp_l = []
    theta_l = []
    N = len(samples_list)
    for k in range(N):
        sum = 0 + 0j
        for n in range(N):
            theta = 2 * np.pi * k * n / N
            sum += samples_list[n] * (math.cos(theta) - 1j * math.sin(theta))
        r, th = cmath.polar(sum)
        amp_l.append(r)
        theta_l.append(th)
    return amp_l, theta_l


def IDFT(amp_l, theta_l):
    samples = []
    N = len(amp_l)
    x = []
    for i in range(N):
        x.append(cmath.rect(amp_l[i], theta_l[i]))
    for n in range(N):
        sum = 0 + 0j
        for k in range(N):
            theta = 2 * np.pi * k * n / N
            sum += x[k] * (math.cos(theta) + 1j * math.sin(theta))
        sum /= N
        samples.append(round(sum.real))
    return samples


def remove_dc_component(samples_list, out_in_freq):
    # Calculate the mean
    mean = sum(samples_list) / len(samples_list)
    new_samples = [x - mean for x in samples_list]
    if out_in_freq:
        new_amp, new_theta = DFT(new_samples)
        return new_amp, new_theta
    else:
        return new_samples


def r12(input1, input2):
    output = []
    N = max(len(input1), len(input2))
    for i in range(len(input1), N):
        input1.append(0)
    for i in range(len(input2), N):
        input2.append(0)

    for j in range(N):
        output.append(0)
        for n in range(N):
            output[j] += input1[n] * input2[(n + j) % N]
        output[j] /= N
    return output


def normalizing_r12(input1, input2):
    output = r12(input1, input2)
    sum1 = 0.0
    sum2 = 0.0
    N = len(output)
    for j in range(N):
        sum1 += input1[j] * input1[j]
        sum2 += input2[j] * input2[j]
    dom = np.sqrt(sum1 * sum2) / N
    x = np.arange(N)
    for j in range(N):
        output[j] = output[j] / dom

    return x, output


def normalize_signal(signal, a, b):
    min_val = min(signal)
    max_val = max(signal)
    # Normalize the signal between a and b
    normalized_signal = [(x - min_val) / (max_val - min_val) * (b - a) + a for x in signal]
    return normalized_signal


def DCT(sample_l):
    dct_list = []
    N = len(sample_l)
    constant = math.sqrt(2 / N)
    pi_over_4N = (np.pi / (4 * N))
    for k in range(N):
        sum = 0
        for n in range(N):
            sum += sample_l[n] * math.cos(pi_over_4N * (2 * n - 1) * (2 * k - 1))
        dct_list.append(constant * sum)
    return dct_list


class PlotPage:
    def __init__(self, notebook, idx):
        self.notebook = notebook
        self.num_plots_columns = 2
        self.num_plots_rows = 1
        self.type = ''
        if idx == 0:
            self.type = 'Smoothing'
        elif idx == 1:
            self.type = 'Sharping'
            self.num_plots_columns = 3
        elif idx == 2:
            self.type = 'Delaying/advancing/folding'
        elif idx == 3:
            self.type = 'Remove the DC component in frequency domain'
            self.num_plots_rows = 3
        elif idx == 4:
            self.type = 'convolve two signals'
            self.num_plots_columns = 3
        elif idx == 5:
            self.type = 'normalized cross-correlation of two signals'
            self.num_plots_columns = 3
        elif idx == 6:
            self.type = 'Time Analysis'
            self.num_plots_columns = 3
        elif idx == 7:
            self.type = 'Template Matching'
            self.num_plots_columns = 0
            self.num_plots_rows = 0
        elif idx == 8:
            self.type = 'Fast convolution'
            self.num_plots_columns = 3
        elif idx == 9:
            self.type = 'Fast correlation'
            self.num_plots_columns = 3
        elif idx == 10:
            self.type = 'Filtering'
            self.num_plots_columns = 3
        elif idx == 11:
            self.type = 'Resampling'
        elif idx == 12:
            self.type = 'distinguish between two subjects'
            self.num_plots_columns = 1
            self.num_plots_rows = 5

        self.frame = tk.Frame(notebook)
        self.frame.pack()
        notebook.add(self.frame, text=f"{self.type}")

        # self.my_canvas = Canvas(self.frame)
        # self.my_canvas.pack(side=LEFT, fill=BOTH, expand=1)
        # self.my_scrollbar = ttk.Scrollbar(self.frame, orient=VERTICAL, command=self.my_canvas.yview)
        # self.my_scrollbar.pack(side = RIGHT,fill="y")
        # self.my_canvas.configure(yscrollcommand=self.my_scrollbar.set)
        # self.my_canvas.bind('<Configure>',lambda e:self.my_canvas.configure(scrollregion=self.my_canvas.bbox("all")))
        # self.center_frame = ttk.Frame(self.my_canvas)
        # self.my_canvas.create_window( (0,0),window=self.center_frame,anchor='nw')
        self.center_frame = ttk.Frame(self.frame)
        self.center_frame.pack()

        self.buttons = []
        self.constants = []
        self.check_box = []
        self.data_list = []
        self.file_path_var = []
        self.browse_buttons = []
        self.objects = []
        self.display = None
        self.canvas = None
        self.current_row = 0

        big_title_label = ttk.Label(self.center_frame, text=self.type, font=("Arial", 24))
        big_title_label.grid(row=self.current_row, column=1)

        self.current_row += 1

        # Example for figure
        if self.num_plots_rows != 0 and self.num_plots_columns != 0:
            if self.type == 'Remove the DC component in frequency domain':
                self.fig, self.axs = plt.subplots(self.num_plots_rows, self.num_plots_columns, figsize=(20, 9))
                plt.subplots_adjust(hspace=1, wspace=0.5)
            else:
                self.fig, self.axs = plt.subplots(self.num_plots_rows, self.num_plots_columns, figsize=(18, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)

        # constant
        if self.type == 'Smoothing':
            self.data_list.append([])
            self.add_input_signal("Signal", "Time", "Amplitude", 0)
            self.add_constant("number of points included in averaging: ")
            self.output = []
            self.display = ttk.Button(self.center_frame, text=f"Display",
                                      command=lambda: self.smoothing(self.data_list[0], self.output,
                                                                     int(self.constants[1].get())))

        elif self.type == 'Sharping':
            FirstDrev = []
            SecondDrev = []
            for i in range(1, len(InputSignal)):
                FirstDrev.append(InputSignal[i] - InputSignal[i - 1])
            for i in range(1, len(InputSignal) - 1):
                SecondDrev.append(InputSignal[i + 1] + InputSignal[i - 1] - 2 * InputSignal[i])
            self.plot_x_and_y(self.axs[0], np.arange(len(InputSignal)), 'Time', InputSignal, 'Distance', 'Input Signal',
                              0)
            self.plot_x_and_y(self.axs[1], np.arange(len(FirstDrev)), 'Time', FirstDrev, 'velocity', '1st derivative',
                              0)
            self.plot_x_and_y(self.axs[2], np.arange(len(SecondDrev)), 'Time', SecondDrev, 'acceleration',
                              '2nd derivative', 0)
            # DerivativeSignal(FirstDrev, SecondDrev)

        elif self.type == 'Delaying/advancing/folding':
            self.data_list.append({})
            self.add_input_signal("Signal", "Time", "Amplitude", 0)
            self.add_constant("shift by: ")
            self.add_constant("Is folded: ")

            self.output = {}
            self.display = ttk.Button(self.center_frame, text=f"Display",
                                      command=lambda: self.sheft_and_folding(self.data_list[0], self.output,
                                                                             float(self.constants[1].get()),
                                                                             float(self.constants[3].get())))

        elif self.type == 'Remove the DC component in frequency domain':
            self.data_list.append([])
            self.add_input_signal("Signal", "Time", "Amplitude", 0)

        elif self.type == 'convolve two signals':
            self.data_list.append({})
            self.add_input_signal("Signal 1", "Time", "Amplitude", 0)
            self.data_list.append({})
            self.add_input_signal("Signal 2", "Time", "Amplitude", 1)

            self.output = {}
            self.display = ttk.Button(self.center_frame, text=f"Display",
                                      command=lambda: self.convolve_two_signals(self.data_list[0], self.data_list[1],
                                                                                self.output))
        elif self.type == 'normalized cross-correlation of two signals':
            self.data_list.append([])
            self.add_input_signal("Signal 1", "Time", "Amplitude", 0)
            self.data_list.append([])
            self.add_input_signal("Signal 2", "Time", "Amplitude", 1)

            self.output = []
            self.display = ttk.Button(self.center_frame, text=f"Display",
                                      command=lambda: self.normalizing_r12(self.data_list[0], self.data_list[1],
                                                                           self.output))
        elif self.type == 'Time Analysis':
            self.data_list.append([])
            self.add_input_signal("Signal 1", "Time", "Amplitude", 0)
            self.data_list.append([])
            self.add_input_signal("Signal 2", "Time", "Amplitude", 1)
            self.add_constant("FS: \n The sampling period")

            self.output = []

            self.display = ttk.Button(self.center_frame, text=f"Display",
                                      command=lambda: self.display_time_analysis(self.data_list[0], self.data_list[1],
                                                                                 self.output))

        elif self.type == 'Template Matching':
            # data_list[Folder][list][element]
            classes = 2
            self.class_signals_name = []
            for i in range(classes):
                self.class_signals_name.append([])
                self.data_list.append([])
                self.add_folder_signals(f'Folder of Class {i + 1}', i, False)
            self.output = []
            self.test_signals_name = []
            self.add_folder_signals(f'Folder of Tests', classes, True)
            self.display = ttk.Button(self.center_frame, text=f"Display",
                                      command=lambda: self.display_template_matching(self.data_list[0],
                                                                                     self.data_list[1], self.output))
            self.headers = ['Test file name', 'Class']
            self.classes_table = ttk.Treeview(self.center_frame, columns=self.headers, show="headings")
            for i in range(len(self.headers)):
                self.classes_table.heading(i, text=self.headers[i])
                self.classes_table.column(i, width=100)
            self.classes_table.grid(row=self.current_row, column=1, padx=5, pady=5)
            self.classes_table.grid_remove()
        elif self.type == 'Fast convolution':
            self.data_list.append({})
            self.add_input_signal("Signal 1", "Time", "Amplitude", 0)
            self.data_list.append({})
            self.add_input_signal("Signal 2", "Time", "Amplitude", 1)
            self.output = {}
            self.display = ttk.Button(self.center_frame, text=f"Display",
                                      command=lambda: self.fast_convolution(self.data_list[0], self.data_list[1],
                                                                            self.output))
        elif self.type == 'Fast correlation':
            self.data_list.append([])
            self.add_input_signal("Signal 1", "Time", "Amplitude", 0)
            self.data_list.append([])
            self.add_input_signal("Signal 2", "Time", "Amplitude", 1)
            self.output = []
            self.display = ttk.Button(self.center_frame, text=f"Display",
                                      command=lambda: self.fast_correlation(self.data_list[0], self.data_list[1],
                                                                            self.output))
        elif self.type == 'Filtering':
            self.add_filter_inputs(['Low pass', 'High pass', 'Band pass', 'Band stop'])
            self.output = []
            self.data_list.append({})
            self.add_input_signal("Signal", "Time", "Amplitude", 0)
            self.display = ttk.Button(self.center_frame, text=f"Display", command=self.display_corresponding_filter)

        elif self.type == 'Resampling':
            self.add_filter_inputs(['Low pass'])
            self.add_constant(' The interpolation  factor L ')
            self.L_entry = self.objects[-1]
            self.add_constant(' The decimation factor M ')
            self.M_entry = self.objects[-1]
            self.data_list.append({})
            self.add_input_signal("Signal", "Time", "Amplitude", 0)
            self.display = ttk.Button(self.center_frame, text=f"Display", command=self.resampling)
        elif self.type == 'distinguish between two subjects':
            self.add_filter_inputs(['Band pass', 'Band stop'])

            self.add_constant(' New Fs (Hz) ')
            self.New_fs_entry = self.objects[-1]
            self.class_signals_name = []
            self.class_signals_name.append([])
            self.data_list.append([])
            self.add_folder_signals('Folder of class 1', 0, 0)

            self.class_signals_name.append([])
            self.data_list.append([])
            self.add_folder_signals('Folder of class 2', 1, 0)

            self.output = []
            self.test_signals_name = []
            self.add_folder_signals('Folder of Tests', 2, 1)
            self.headers = ['Test file name', 'Class']
            self.classes_table = ttk.Treeview(self.center_frame, columns=self.headers, show="headings")
            for i in range(len(self.headers)):
                self.classes_table.heading(i, text=self.headers[i])
                self.classes_table.column(i, width=100)
            self.classes_table.grid(row=self.current_row, column=1, padx=5, pady=5)
            self.classes_table.grid_remove()
            self.current_row += 1
            self.display = ttk.Button(self.center_frame, text=f"Display", command=self.distinguish_between_two_subjects)
            self.steps_of_preprocessing = ttk.Button(self.center_frame, text=f"Show steps of preprocessing",
                                                     command=self.open_foler_in_explorer)

        # for object in self.objects:
        #     object.pack()

        if self.display != None:
            self.current_row += 1
            self.display.grid(row=self.current_row, column=1, padx=5, pady=5)
            self.current_row += 1

    def add_constant(self, details):
        self.objects.append(ttk.Label(self.center_frame, text=details))
        self.objects[-1].grid(row=self.current_row, column=0, padx=5, pady=5)
        self.constants.append(ttk.Entry(self.center_frame))
        self.objects.append(self.constants[-1])
        self.objects[-1].grid(row=self.current_row, column=1, padx=5, pady=5)
        self.current_row += 1

    def add_filter_inputs(self, filter_list):
        self.filter_type = tk.StringVar()
        self.filter_label = ttk.Label(self.center_frame, text="Filter Type")
        self.filter_label.grid(row=self.current_row, column=0, padx=5, pady=5)
        filters = filter_list  # any of this ['Low pass', 'High pass', 'Band pass', 'Band stop']
        self.Filter_combobox = ttk.Combobox(self.center_frame, textvariable=self.filter_type,
                                            values=filters, state='readonly')
        self.Filter_combobox.grid(row=self.current_row, column=1, padx=5, pady=5)
        self.load_spec = ttk.Button(self.center_frame, text='Load Filter specifications', command=self.read_filter_file)
        self.load_spec.grid(row=self.current_row, column=2, padx=5, pady=5)
        def filter_type(event):
            if self.Filter_combobox.get() in ['Low pass', 'High pass']:
                self.f1_entry.grid(column=1)
                self.f1_label.grid_remove()
                self.f2_label.grid_remove()
                self.f2_entry.grid_remove()
            else:
                self.f1_label.grid()
                self.f1_entry.grid(column=2)
                self.f2_label.grid()
                self.f2_entry.grid()

        self.change_filter_type = filter_type
        self.Filter_combobox.bind("<<ComboboxSelected>>", self.change_filter_type)
        self.current_row += 1

        self.add_constant('Sampling Frequency Fs (Hz)')
        self.fs_entry = self.objects[-1]
        self.f_title = ttk.Label(self.center_frame, text='Cutting Frequency FC (Hz)')
        self.f1_label = ttk.Label(self.center_frame, text='Minimum')
        self.f1_entry = ttk.Entry(self.center_frame)
        self.f2_label = ttk.Label(self.center_frame, text='Maximum')
        self.f2_entry = ttk.Entry(self.center_frame)

        self.f_title.grid(row=self.current_row, column=0, padx=5, pady=5)
        self.f1_label.grid(row=self.current_row, column=1, padx=5, pady=5)
        self.f1_entry.grid(row=self.current_row, column=1, padx=5, pady=5)
        self.f2_label.grid(row=self.current_row, column=3, padx=5, pady=5)
        self.f2_entry.grid(row=self.current_row, column=4, padx=5, pady=5)
        self.f1_label.grid_remove()
        self.f2_label.grid_remove()
        self.f2_entry.grid_remove()
        self.current_row += 1

        self.add_constant('Stop attenuation δs (DB)')
        self.stop_attenuation_entry = self.objects[-1]

        self.add_constant('Transition width △F (HZ) ')
        self.Transition_width_entry = self.objects[-1]
        self.Filter_combobox.current(0)
        self.change_filter_type(0)

    def read_filter_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        filter_params = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                filter_params[key.strip()] = value.strip()
        # Accessing the parameters
        self.Filter_combobox.set(filter_params['FilterType'])
        self.change_filter_type(0)
        self.fs_entry.delete(0,tk.END)
        self.fs_entry.insert(0,filter_params['FS'])
        self.stop_attenuation_entry.delete(0,tk.END)
        self.stop_attenuation_entry.insert(0,filter_params['StopBandAttenuation'])
        self.Transition_width_entry.delete(0,tk.END)
        self.Transition_width_entry.insert(0,filter_params['TransitionBand'])
        if filter_params['FilterType'] in ['Low pass','High pass']:
            self.f1_entry.delete(0, tk.END)
            self.f1_entry.insert(0, filter_params['FC'])
        else:
            self.f1_entry.delete(0, tk.END)
            self.f1_entry.insert(0, filter_params['F1'])

            self.f2_entry.delete(0, tk.END)
            self.f2_entry.insert(0, filter_params['F2'])

    def add_input_signal(self, detials, xlabel, ylabel, i):
        self.objects.append(ttk.Label(self.center_frame, text=f'{detials} path'))
        self.objects[-1].grid(row=self.current_row, column=0, padx=5, pady=5)
        self.file_path_var.append(tk.StringVar())
        self.objects.append(
            ttk.Entry(self.center_frame, textvariable=self.file_path_var[-1], width=50, state='readonly'))
        self.objects[-1].grid(row=self.current_row, column=1, padx=5, pady=5)
        self.browse_buttons.append(
            ttk.Button(self.center_frame, text='Browse', command=lambda: self.read_file_into_data(i, xlabel, ylabel)))
        self.objects.append(self.browse_buttons[-1])
        self.objects[-1].grid(row=self.current_row, column=2, padx=5, pady=5)
        self.current_row += 1

    def add_folder_signals(self, detials, i, is_test):
        self.objects.append(ttk.Label(self.center_frame, text=f'{detials} path'))
        self.objects[-1].grid(row=self.current_row, column=0, padx=5, pady=5)
        self.file_path_var.append(tk.StringVar())
        self.objects.append(ttk.Entry(self.center_frame, textvariable=self.file_path_var[-1], width=50))
        self.objects[-1].grid(row=self.current_row, column=1, padx=5, pady=5)
        self.browse_buttons.append(
            ttk.Button(self.center_frame, text='Browse', command=lambda: self.browse_signals_folder(i, is_test)))
        self.objects.append(self.browse_buttons[-1])
        self.objects[-1].grid(row=self.current_row, column=2, padx=5, pady=5)
        self.current_row += 1

    def set_ax(self, ax, x, xlabel, y, ylabel, title, discrete):
        try:
            ax.clear()
            x = [int(idx) for idx in x]
            if discrete:
                ax.stem(x, y)
            else:
                ax.plot(x, y)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        except Exception as e:
            messagebox.showerror("Error", f"message : {str(e)}")

    def plot_x_and_y(self, ax, x, xlabel, y, ylabel, title, discrete):
        try:
            ax.clear()
            x = [int(idx) for idx in x]
            if discrete:
                ax.stem(x, y)
            else:
                ax.plot(x, y)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if self.canvas != None:
                self.canvas.get_tk_widget().pack()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"message : {str(e)}")

    def read_file_into_data(self, idx, xlabel, ylabel):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        self.file_path_var[idx].set(file_path)
        self.data_list[idx].clear()
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines[3:]:
                    x, y = line.split()
                    if isinstance(self.data_list[idx], dict):
                        self.data_list[idx][int(x)] = float(y)
                    else:
                        self.data_list[idx].append(float(y))
        except Exception as e:
            messagebox.showerror("Error", f"message : {str(e)}")

        if self.type == 'Remove the DC component in frequency domain':
            self.omega = []
            samples = self.data_list[idx]
            Amp, Theta = DFT(samples)
            n = len(samples)
            self.omega.append(2 * np.pi / n * 4000)
            for i in range(1, n):
                self.omega.append(self.omega[0] + self.omega[i - 1])
            self.plot_x_and_y(self.axs[0][0], np.arange(n), 'index(n)', samples, 'Amplitude(A)',
                              f'output Signal in Time Domain', 0)
            self.plot_x_and_y(self.axs[1][0], self.omega, 'Frequency(HZ)', Amp, 'Amplitude(A)',
                              'Input Signal in Frequency Domain', 1)
            self.plot_x_and_y(self.axs[2][0], self.omega, 'Frequency(HZ)', Theta, 'Theta(θ)',
                              'Input Signal in Frequency Domain', 1)
            Amp[0] = 0
            Theta[0] = 0
            new_samples = IDFT(Amp, Theta)
            n = len(new_samples)
            self.omega = []
            self.omega.append(2 * np.pi / n * 4000)
            for i in range(1, n):
                self.omega.append(self.omega[0] + self.omega[i - 1])
            self.plot_x_and_y(self.axs[0][1], np.arange(n), 'index(n)', new_samples, 'Amplitude(A)',
                              f'output Signal in Time Domain', 0)
            self.plot_x_and_y(self.axs[1][1], self.omega, 'Frequency(HZ)', Amp, 'Amplitude(A)',
                              'Output Signal in Frequency Domain', 1)
            self.plot_x_and_y(self.axs[2][1], self.omega, 'Frequency(HZ)', Theta, 'Theta(θ)',
                              'Output Signal in Frequency Domain', 1)
            SignalSamplesAreEqual("Remove DC component/DC_component_output.txt", [round(x, 3) for x in new_samples])

        elif isinstance(self.data_list[idx], dict):
            x = list(self.data_list[idx].keys())
            y = list(self.data_list[idx].values())
            self.plot_x_and_y(self.axs[idx], x, xlabel, y, ylabel, f'Signal {idx + 1}',
                              self.type not in ['Delaying/advancing/folding',
                                                'normalized cross-correlation of two signals', 'Filtering',
                                                'Resampling'])
        else:
            self.plot_x_and_y(self.axs[idx], np.arange(len(self.data_list[idx])), xlabel, self.data_list[idx], ylabel,
                              f'Signal {idx + 1}', 0)

    def browse_signals_folder(self, idx, isTestFolder):
        folder_path = filedialog.askdirectory(title="Select Folder")
        self.file_path_var[idx].set(folder_path)
        if folder_path:
            if isTestFolder:
                self.output.clear()
                self.test_signals_name.clear()
            else:
                self.data_list[idx].clear()
                self.class_signals_name[idx].clear()
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if filename.endswith('.txt') and os.path.isfile(file_path):
                    with open(file_path, 'r') as file:
                        tmp = []
                        for line in file:
                            tmp.append(float(line))
                    base_name, extension = os.path.splitext(filename)
                    if isTestFolder:
                        self.output.append(tmp)
                        self.test_signals_name.append(base_name)
                    else:
                        self.data_list[idx].append(tmp)
                        self.class_signals_name[idx].append(base_name)

    def smoothing(self, input, output, k):
        try:
            # equation
            # y[i] = 1/M sum j = 0 to M-1:x[i+j]
            output.clear()
            N = len(input)
            pre = [input[0]]
            for i in range(1, N):
                pre.append(pre[i - 1] + input[i])
            for i in range(N - k):
                # N-k-1
                # Scenario 1
                mn = max(0, i - 1)
                mx = min(i + k, N - 1)
                # Scenario 2
                # mn =max(i-k//2-1,0)
                # mx = min(i+k//2,N-1)
                if mn == 0:
                    output.append(input[0])
                else:
                    output.append(0)
                output[i] += (pre[mx] - pre[mn])
                # output.append(0)
                # for j in range(k):
                #     output[i]+=float(input[i+j])

                output[i] /= k
            self.plot_x_and_y(self.axs[-1], np.arange(len(output)), 'Time', output, 'Amplitude',
                              'Output after smoothing', 0)
        except Exception as e:
            messagebox.showerror("Error", f" smoothing: 0<Window size<N : {str(e)}")

    def sheft_and_folding(self, input, output, k, is_folding):
        output.clear()
        sign = 1
        titel = 'The signal has been '
        if is_folding != 0:
            sign = -1
            titel += 'folded and '
        titel += f'shifted by {k}'
        # output[i] = input[sign*i+k]
        # sign*i + k = min_key -> i = (min_key - k)*sign , (max_key - k ) *sign
        st = (int(min(input)) - k) * sign
        en = (int(max(input)) - k) * sign
        if st > en:
            st, en = en, st
        st = int(st)
        en = int(en)
        for i in range(st, en + 1):
            output[i] = input.get(sign * i + k, 0)
        indices = list(output.keys())
        indices = [int(num) for num in indices]
        samples = list(output.values())
        samples = [int(num) for num in samples]
        self.plot_x_and_y(self.axs[-1], indices, 'index', samples, 'value', titel, 0)
        generate_file(titel, 0, 0, indices, samples)

        if is_folding != 0:
            if k == 0:
                Shift_Fold_Signal("TestCases/Shifting and Folding/input_fold.txt", indices, samples)
            elif k == 500:
                Shift_Fold_Signal("TestCases/Shifting and Folding/Output_ShifFoldedby500.txt", indices, samples)
            elif k == -500:
                Shift_Fold_Signal("TestCases/Shifting and Folding/Output_ShiftFoldedby-500.txt", indices, samples)

    def convolve_signals(self, input1, input2):
        st1 = min(input1)
        en1 = max(input1)
        st2 = min(input2)
        en2 = max(input2)
        mn = min(st1, st2)
        mx = max(en1, en2)
        output = {}
        for i in range(st1 + st2, en1 + en2 + 1):
            output[i] = 0.0
            for k in range(mn, min(en1,i-st2)):
                if k > en1 or  k > i - st2:  # k > min(en1,i-st2)
                    break
                output[i] += (input1.get(k, 0.0) * input2.get(i - k, 0.0))
        x = list(output.keys())
        y = list(output.values())
        while y[-1] == 0:
            x.pop(-1), y.pop(-1)
        while y[0] == 0:
            x.pop(0), y.pop(0)
        return x, y

    def convolve_two_signals(self, input1, input2, output):
        x, y = self.convolve_signals(input1, input2)
        generate_file("convolution of two signals", 0, 0, x, y)
        self.plot_x_and_y(self.axs[-1], x, 'index', y, 'value', f'Output of convolve the two Signal', 1)
        ConvTest(x, y)

    # TASK 7
    def r12(self, input1, input2, output):
        output.clear()
        N = max(len(input1), len(input2))
        for i in range(len(input1), N):
            input1.append(0)
        for i in range(len(input2), N):
            input2.append(0)

        for j in range(N):
            output.append(0)
            for n in range(N):
                output[j] += input1[n] * input2[(n + j) % N]
            output[j] /= N

    def normalize_correlation(self, input1, input2, output):
        sum1 = 0.0
        sum2 = 0.0
        N = len(output)
        for j in range(N):
            sum1 += input1[j] * input1[j]
            sum2 += input2[j] * input2[j]
        dom = np.sqrt(sum1 * sum2) / N
        for j in range(N):
            output[j] = output[j] / dom

    def normalizing_r12(self, input1, input2, output):
        self.r12(input1, input2, output)
        self.normalize_correlation(input1, input2, output)
        x = np.arange(N)
        self.plot_x_and_y(self.axs[-1], x, 'time', output, 'amp', f'Normalizing Cross-Correlation of the two Signal', 0)
        Compare_Signals('task7/Task Files/Point1 Correlation/CorrOutput.txt', x, output)

    def display_time_analysis(self, input1, input2, output):
        self.r12(input1, input2, output)
        mx_lag = np.argmax(np.abs(output))
        self.plot_x_and_y(self.axs[-1], np.arange(len(output)), 'time', output, 'amp',
                          f'Cross-Correlation of the two Signal', 0)

        self.axs[-1].annotate('Max absolute value', xy=(mx_lag, output[mx_lag]),
                              fontsize=7, xytext=(mx_lag + 1, output[mx_lag]),
                              arrowprops=dict(facecolor='red'),
                              color='g')
        time_delay = mx_lag / float(self.constants[0].get())
        self.axs[-1].text(0.05, 0.9, f'Time Delay: {time_delay}s', fontsize=12, color='purple',
                          transform=self.axs[-1].transAxes)
        self.canvas.draw()

    def display_template_matching(self, data1, data2, tests):
        for row in self.classes_table.get_children():
            self.classes_table.delete(row)

        def correlate_signals(data, test):
            mx = -1
            for signal in data:
                cor = r12(signal, test)
                self.normalize_correlation(signal, test, cor)
                mx = max(mx, max(cor))
            return mx

        i = 0
        for signal_test in tests:
            max_correlation1 = correlate_signals(data1, signal_test)
            max_correlation2 = correlate_signals(data2, signal_test)
            print(max_correlation1, max_correlation2)
            c = 'A' if max_correlation1 > max_correlation2 else 'B'
            self.classes_table.insert("", "end", values=(self.test_signals_name[i], c))
            i += 1
        self.classes_table.grid()

    # Task 8
    def fast_convolution(self, signal1, signal2, output):
        x1 = list(signal1.values())
        x2 = list(signal2.values())
        mn = min(min(signal1), min(signal2))
        L = len(x1) + len(x2) - 1

        def fix_len(x):
            while len(x) < L:
                x.append(0)

        fix_len(x1)
        fix_len(x2)

        print(x1)
        print(x2)

        amp1, theta1 = DFT(x1)
        amp2, theta2 = DFT(x2)

        amp = []
        theta = []
        idx = []
        for i in range(len(x1)):
            r, th = cmath.polar(cmath.rect(amp1[i], theta1[i]) * cmath.rect(amp2[i], theta2[i]))
            amp.append(r)
            theta.append(th)
            idx.append(mn + i)
        output = IDFT(amp, theta)
        print(output)
        self.plot_x_and_y(self.axs[-1], idx, 'index', output, 'value', f'Output of convolve the two Signal', 1)
        ConvTest(idx, output)

    def fast_correlation(self, input1, input2, output):
        amp1, theta1 = DFT(input1)
        amp2, theta2 = DFT(input2)
        output.clear()
        amp = []
        theta = []
        N = len(amp1)
        for i in range(N):
            r, th = cmath.polar(cmath.rect(amp1[i], theta1[i]).conjugate() * cmath.rect(amp2[i], theta2[i]))
            amp.append(r)
            theta.append(th)

        output = IDFT(amp, theta)
        output = [x / N for x in output]
        print(output)
        x = np.arange(len(output))
        self.plot_x_and_y(self.axs[-1], x, 'time', output, 'amp', f'Normalizing Cross-Correlation of the two Signal', 0)
        Compare_Signals('task7/Task Files/Point1 Correlation/Corr_Output.txt', x, output)
        generate_file("Correlation_output", 0, 0, x, output)

    # Task 9
    def hD(self, n, filter_type, f1_normalized, f2_normalized=0):
        if filter_type == 'Low pass':
            if n == 0:
                return 2 * f1_normalized
            else:
                nwc = n * (2 * np.pi * f1_normalized)
                return 2 * f1_normalized * cmath.sin(nwc) / nwc
        elif filter_type == 'High pass':
            if n == 0:
                return 1 - 2 * f1_normalized
            else:
                nwc = n * (2 * np.pi * f1_normalized)
                return -2 * f1_normalized * cmath.sin(nwc) / nwc
        elif filter_type == 'Band pass':
            if n == 0:
                return 2 * (f2_normalized - f1_normalized)
            else:
                nwc1 = n * (2 * np.pi * f1_normalized)
                nwc2 = n * (2 * np.pi * f2_normalized)
                return -2 * ((f1_normalized * cmath.sin(nwc1) / nwc1) - f2_normalized * cmath.sin(nwc2) / nwc2)
        elif filter_type == 'Band stop':
            if n == 0:
                return 1 - 2 * (f2_normalized - f1_normalized)
            else:
                nwc1 = n * (2 * np.pi * f1_normalized)
                nwc2 = n * (2 * np.pi * f2_normalized)
                return 2 * ((f1_normalized * cmath.sin(nwc1) / nwc1) - f2_normalized * cmath.sin(nwc2) / nwc2)
        else:
            print("\n INVALID PARAMETER FUNCTION: hD \n")
        return 0

    def window_funtion(self, n, N, name):
        if name == 'Rectangular':
            return 1
        elif name == 'Hanning':
            return 0.5 + 0.5 * cmath.cos(2 * np.pi * n / N)
        elif name == 'Hamming':
            return 0.54 + 0.46 * cmath.cos(2 * np.pi * n / N)
        elif name == 'Blackman':
            return 0.42 + 0.5 * cmath.cos(2 * np.pi * n / (N - 1)) + 0.08 * cmath.cos(4 * np.pi * n / (N - 1))
        else:
            print("\n INVALID NAME WINDOW: def window_funtion(self,n,N,name) \n")
        return 0

    def get_window_name(self, stopband_attenuation):
        if stopband_attenuation <= 21:
            return 'Rectangular'
        elif stopband_attenuation <= 44:
            return 'Hanning'
        elif stopband_attenuation <= 53:
            return 'Hamming'
        elif stopband_attenuation <= 74:
            return 'Blackman'
        else:
            print("\n INVALID stopband_attenuation: stopband_attenuation > 74  \n")
        return None

    def calculate_N(self, transition_width_normalized, name):
        N = 0
        C = 0
        if name == 'Rectangular':
            C += 0.9
        elif name == 'Hanning':
            C += 3.1
        elif name == 'Hamming':
            C += 3.3
        elif name == 'Blackman':
            C += 5.5
        else:
            print("\n INVALID NAME WINDOW: def calculate_N(self,transition_width_normalized,name) \n")
        N += math.ceil(C / transition_width_normalized)
        print(f'\ntransition_width_normalized: {transition_width_normalized}\n')
        if (int(N) & 1) == 0:
            N += 1
        return N

    def calculate_cutoff_frequency(self, filter_type, transistion_width_normalized, f1_normalized, f2_normalized):
        if filter_type == 'Low pass':
            return f1_normalized + transistion_width_normalized / 2, 0
        elif filter_type == 'High pass':
            return f1_normalized - transistion_width_normalized / 2, 0
        elif filter_type == 'Band pass':
            return f1_normalized - transistion_width_normalized / 2, f2_normalized + transistion_width_normalized / 2
        elif filter_type == 'Band stop':
            return f1_normalized + transistion_width_normalized / 2, f2_normalized - transistion_width_normalized / 2
        else:
            print("\n INVALID PARAMETER FUNCTION: hD \n")
        return 0, 0

    def get_h_of_filter(self, filter_type, stop_attenuation, transistion_width_normalized, f1_normalized,
                        f2_normalized):
        window_name = self.get_window_name(stop_attenuation)
        self.N_window = N = self.calculate_N(transistion_width_normalized, window_name)
        f1_normalized, f2_normalized = self.calculate_cutoff_frequency(filter_type, transistion_width_normalized,
                                                                       f1_normalized, f2_normalized)
        h = [self.hD(0, filter_type, f1_normalized, f2_normalized)]
        print(f"window name: {window_name}\nN= {N}\n")
        for n in range(1, int((N - 1) / 2 + 1)):
            hw = self.hD(n, filter_type, f1_normalized, f2_normalized) * self.window_funtion(n, N, window_name)
            h.append(hw.real)
            h.insert(0, hw.real)
        return -int((N - 1) / 2), h

    def display_corresponding_filter(self):
        filter_type = self.Filter_combobox.get()
        stop_attenuation = float(self.stop_attenuation_entry.get())
        fs = float(self.fs_entry.get())
        transistion_width_normalized = float(self.Transition_width_entry.get()) / fs
        f1_normalized = float(self.f1_entry.get()) / fs
        f2_normalized = 0
        if bool(self.f2_entry.get()):
            f2_normalized = float(self.f2_entry.get()) / fs

        st, y_h = self.get_h_of_filter(filter_type, stop_attenuation, transistion_width_normalized, f1_normalized,
                                       f2_normalized)
        x_h = np.arange(st, st + len(y_h))
        # generate_file('Cofficient Of' + filter_type, 0, 0, x_h, y_h)
        self.plot_x_and_y(self.axs[1], x_h, 'Time', y_h, 'Amplitude', filter_type + 'Cofficients',
                          0)
        if filter_type == 'Low pass':
            Compare_Signals('FIR test cases/Testcase 1/LPFCoefficients.txt', x_h, y_h)
        elif filter_type == 'High pass':
            Compare_Signals('FIR test cases/Testcase 2/HPFCoefficients.txt', x_h, y_h)
        elif filter_type == 'Band pass':
            Compare_Signals('FIR test cases/Testcase 3/BPFCoefficients.txt', x_h, y_h)
        elif filter_type == 'Band stop':
            Compare_Signals('FIR test cases/Testcase 4/BSFCoefficients.txt', x_h, y_h)

        if bool(self.data_list[0]):
            x, y = self.convolve_signals(self.data_list[0], {key: value for key, value in zip(x_h, y_h)})
            # generate_file(f'Filtered signal by {filter_type}', 0, 0, x, y)
            self.plot_x_and_y(self.axs[-1], x, 'Time', y, 'Amplitude', f'Filtered Signal with {filter_type}', 0)
            if filter_type == 'Low pass':
                Compare_Signals('FIR test cases/Testcase 1/ecg_low_pass_filtered.txt', x, y)
            elif filter_type == 'High pass':
                Compare_Signals('FIR test cases/Testcase 2/ecg_high_pass_filtered.txt', x, y)
            elif filter_type == 'Band pass':
                Compare_Signals('FIR test cases/Testcase 3/ecg_band_pass_filtered.txt', x, y)
            elif filter_type == 'Band stop':
                Compare_Signals('FIR test cases/Testcase 4/ecg_band_stop_filtered.txt', x, y)

    def filter_signal(self, samples, from_input):
        if not from_input:
            filter_type = 'Low pass'
            fs = 8000
            stop_attenuation = 50
            f1_normalized = 1500 / fs
            transistion_width_normalized = 500 / fs
            f2_normalized = 0
        # apply Low pass filter
        else:
            filter_type = self.Filter_combobox.get()
            stop_attenuation = float(self.stop_attenuation_entry.get())
            fs = float(self.fs_entry.get())
            transistion_width_normalized = float(self.Transition_width_entry.get()) / fs
            f1_normalized = float(self.f1_entry.get()) / fs
            f2_normalized = 0
            if bool(self.f2_entry.get()):
                f2_normalized = float(self.f2_entry.get()) / fs
        st, y_h = self.get_h_of_filter(filter_type, stop_attenuation, transistion_width_normalized, f1_normalized,
                                       f2_normalized)
        x_h = np.arange(st, st + len(y_h))
        x, y = self.convolve_signals(samples, {key: value for key, value in zip(x_h, y_h)})
        return x, y

    def resapmleby_l_over_M(self, samples, L, M, filter_from_input):
        if L != 0 and M != 0:
            gc = math.gcd(int(L), int(M))
            L /= gc
            M /= gc
        up_sampling = {}
        if L != 0:
            # new_samples = x(n/L)
            # (N-1)/L = n-1 -> Last_point
            # N = (n-1)*L+1
            N = (len(samples) - 1) * L + 1
            st = min(samples)
            y = list(samples.values())
            for i in range(int(N)):
                if i % L == 0:
                    up_sampling[st + i] = y[int(i / L)]
                else:
                    up_sampling[st + i] = 0
        else:
            up_sampling = self.data_list[0]
        x, y = self.filter_signal(up_sampling, filter_from_input)
        final_ans = []
        if M != 0:  # y(n) = x(n*M)
            N = int((len(y) - 1) / M) + 1
            # (N-1) = x*M
            st = x[0]
            for n in range(int(N)):
                final_ans.append(y[int(n * M)])
            x = np.arange(st, st + N)
        else:
            final_ans = y
        return x, final_ans

    def resampling(self):
        L = 0  # factor of up sampling
        M = 0  # factor of down sampling
        if bool(self.L_entry.get()):
            L = float(self.L_entry.get())
        if bool(self.M_entry.get()):
            M = float(self.M_entry.get())
        if (L == 0 and M == 0) or min(L, M) < 0 or L != int(L) or M != int(M):
            messagebox.showerror("Error",
                                 "M and L should be non-negative integer values and at least one of them not equal to zero")
            return
        filter_type = self.Filter_combobox.get()
        stop_attenuation = float(self.stop_attenuation_entry.get())
        fs = float(self.fs_entry.get())
        transistion_width_normalized = float(self.Transition_width_entry.get()) / fs
        f1_normalized = float(self.f1_entry.get()) / fs
        f2_normalized = 0
        if bool(self.f2_entry.get()):
            f2_normalized = float(self.f2_entry.get()) / fs

        x, final_ans = self.resapmleby_l_over_M(self.data_list[0], L, M, 1)
        self.plot_x_and_y(self.axs[-1], x, 'Time', final_ans, 'Amplitude',
                          f'Upsampling by {L} factor and down sampling by {M} factor', 0)
        generate_file(f'Upsampling by {L} factor and down sampling by {M} factor', 0, 0, x, final_ans)
        if L == 0 and M == 2:
            Compare_Signals('Sampling test cases/Testcase 1/Sampling_Down.txt', x, final_ans)
        elif L == 3 and M == 0:
            Compare_Signals('Sampling test cases/Testcase 2/Sampling_Up.txt', x, final_ans)
        elif L == 3 and M == 2:
            Compare_Signals('Sampling test cases/Testcase 3/Sampling_Up_Down.txt', x, final_ans)

    def open_foler_in_explorer(self):
        subprocess.run(['explorer', 'Data After Preprocessing'])  # For Windows

    def distinguish_between_two_subjects(self):
        if bool(self.New_fs_entry.get()):
            new_fs = float(self.New_fs_entry.get())
        else:
            new_fs = 0
        fs = float(self.fs_entry.get())
        resampling = 0
        # check resampling
        if new_fs >= 2 * float(self.f2_entry.get()):
            resampling = 1
        else:
            messagebox.showerror("Error", "newFs is not valid")

        def plot(folder_path, file_name, x, y, title):
            plt.figure()
            plt.plot(x, y)
            plt.title(f'{title}')
            plt.xlabel('Time(s)')
            plt.ylabel('Amplitude')
            file_path = os.path.join(folder_path + '/Plots', file_name+'.png')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            plt.savefig(file_path)
            plt.close()
            file_path = os.path.join(folder_path + '/Files', file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            generate_file(file_path, 0, 1, x, y)

        def pre_processing(folder_name, data_set, names):
            i = 0
            print(names)
            for signal in data_set:
                # Specify the folder path
                main_folder = f'Data After Preprocessing/{folder_name}'
                folder_path = os.path.join(main_folder, names[i])
                # Clear the folder if it exists
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                # Create the nested folder to save plots
                os.makedirs(folder_path, exist_ok=True)
                #########################################################################################################
                # orignal signal
                x = np.arange(0, len(signal))
                plot(folder_path, '0- Original', x, signal, 'Original signal')
                #########################################################################################################
                # 1- Filter the signal using FIR filter with band [miniF, maxF].
                x, signal = self.filter_signal({key: value for key, value in zip(x, signal)}, 1)
                plot(folder_path, f'1- filtering', x, signal, 'Filtered signal')
                #########################################################################################################
                j = 2
                #########################################################################################################
                # 2- Resample the signal to newFs
                if resampling:
                    # resampling by new_fs / old_fs factor  we can say L = new_fs w M = old_fs
                    x, signal = self.resapmleby_l_over_M({key: value for key, value in zip(x, signal)}, new_fs, fs, 0)
                    plot(folder_path, '2- resampling', x, signal, 'Signal after resampling')
                    j += 1
                #########################################################################################################
                # 3- Remove the DC component.
                mean = sum(signal) / len(signal)
                signal = [x - mean for x in signal]
                plot(folder_path, f'{j}- Remove the DC component', x, signal, 'Signal after removing Dc')
                j += 1
                #########################################################################################################
                # 4- Normalize the signal to be from -1 to 1.
                signal = normalize_signal(signal, -1, 1)
                plot(folder_path, f'{j}- Normalize the signal to be from -1 to 1.png', x, signal,
                     'Signal after normalizing from -1 to 1')
                j += 1
                #########################################################################################################
                # 5- Compute Auto correlation for each ECG segment
                signal = r12(signal, signal)
                plot(folder_path, f'{j}- Auto correlation', x, signal, 'Signal after Auto correlation')
                j += 1
                #########################################################################################################
                # 6- Preserve only the needed coefficients for the computed auto correlation.
                idx_of_max_value = 0
                for k in range(len(signal)):
                    if signal[k] > signal[idx_of_max_value]:
                        idx_of_max_value = k
                new_signal = []
                x = []
                # samples      time
                # fs           1s
                # N         ~avg = 200ms=0.2s   half of heartbeat Time
                N = 0.2 * (new_fs if resampling else fs)
                for k in range(idx_of_max_value, min(int(idx_of_max_value + N), len(signal))):
                    new_signal.append(signal[k])
                    x.append(k - idx_of_max_value)
                signal = new_signal
                plot(folder_path, f'{j}- Preserve only the needed coefficients', x, signal,
                     'Signal after Preserving only the needed coefficients')
                j += 1
                #########################################################################################################
                # 7- Compute DCT
                signal = DCT(signal)
                # removing zeros from DCT
                signal = [val for val in signal if abs(val) >= 0.01]
                x = np.arange(0, len(signal))
                plot(folder_path, f'{j}- Computing DCT', x, signal, 'Signal after Computing DCT')
                j += 1
                #########################################################################################################
                i += 1

        pre_processing('Class A signals', self.data_list[0], self.class_signals_name[0])
        pre_processing('Class B signals', self.data_list[1], self.class_signals_name[1])
        pre_processing('Tests signals', self.output, self.test_signals_name)
        self.display_template_matching(self.data_list[0], self.data_list[1], self.output)
        self.steps_of_preprocessing.grid(row=self.current_row, column=1, padx=5, pady=5)


class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Plotter")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=2, fill='both',ipadx=10,ipady=10)

        self.pages = []
        for i in range(13):
            page = PlotPage(self.notebook, i)
            self.pages.append(page)

        self.show_page(0)

    def show_page(self, page_num):
        self.notebook.select(page_num)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()
