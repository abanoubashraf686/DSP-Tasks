import tkinter as tk
from tkinter import ttk
from tkinter import filedialog,messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
class PlotPage:
    def __init__(self, notebook, idx):
        self.notebook = notebook
        self.type = ''
        if idx == 0:
            self.type = 'add'
        elif idx == 1:
            self.type = 'sub'
        elif idx == 2:
            self.type = 'multiply'
        elif idx == 3:
            self.type = 'square'
        elif idx == 4:
            self.type = 'shift'
        elif idx == 5:
            self.type = 'normalize'
        elif idx == 6:
            self.type = 'accumulate'
        elif idx==7:
            self.type = 'generate'
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=f"{self.type}")
        if idx == 7:
            # Create Matplotlib figure and canvas
            self.fig, (self.ax1) = plt.subplots(1, 1, figsize=(1920 / 96, 1080 / 192), dpi=96)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
            canvas_widget = self.canvas.get_tk_widget()
            canvas_widget.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

            # Variable for function choice
            function_choice = tk.StringVar()
            function_choice.set('Sine')

            # Variables for input fields
            arr_of_var = []
            for i in range(5):
                x = tk.StringVar()
                arr_of_var.append(x)
            # Create a big title label at the top
            big_title_label = ttk.Label(self.frame, text="Function Plotter", font=("Arial", 24))
            big_title_label.grid(row=0, column=0, columnspan=3, padx=5, pady=20, sticky='n')

            # Create a frame to center the elements in the middle
            center_frame = ttk.Frame(self.frame)
            center_frame.grid(row=1, column=0, columnspan=3)
            # Label and Combobox for function choice
            function_label = ttk.Label(center_frame, text="Select Function:")
            function_label.grid(row=0, column=0, padx=5, pady=5)
            function_combobox = ttk.Combobox(center_frame, textvariable=function_choice,
                                             values=['Sine', 'Cosine', 'Both'], state='readonly')
            function_combobox.grid(row=0, column=1, padx=5, pady=5)
            arr_of_text = ["Amplitude (A):", "Phase Shift (Î¸) (radians):", "Analog Frequency (F) (Hz):",
                           "Sampling Frequency (Fs):", "Number of cycles in the plot:"]
            arr_of_sliders = []

            for i in range(5):
                label = ttk.Label(center_frame, text=arr_of_text[i])
                label.grid(row=i + 1, column=0, padx=5, pady=5)
                slider = ttk.Scale(center_frame, from_=0, to=10, orient='horizontal', length=200,
                                   variable=arr_of_var[i])
                slider.grid(row=i + 1, column=1, padx=5, pady=5)
                entry = ttk.Entry(center_frame, textvariable=arr_of_var[i])
                entry.grid(row=i + 1, column=2, padx=5, pady=5)
                arr_of_var[i].set(slider.get())
                arr_of_sliders.append(slider)

            # Plot button
            plot_button = ttk.Button(center_frame, text="Plot Signal and generate output files",
                                     command=lambda: self.plot_function(function_choice, arr_of_sliders, self.ax1))
            plot_button.grid(row=6, columnspan=3, padx=5, pady=10)

            # Entry and Browse button for file path
            self.file_path_var = tk.StringVar()
            file_path_entry = ttk.Entry(center_frame, textvariable=self.file_path_var, width=50)
            file_path_entry.grid(row=7, column=0, padx=5, pady=5, columnspan=2, sticky='w')
            browse_button = ttk.Button(center_frame, text="Browse", command=self.browse_file)
            browse_button.grid(row=7, column=2, padx=5, pady=5, sticky='e')

            # Load button for loaded signal
            load_button = ttk.Button(center_frame, text="Load Signal", command=self.plot_loaded_signal)
            load_button.grid(row=8, columnspan=3, padx=5, pady=10)

            # Bind events to update entry fields
            for i in range(5):
                arr_of_sliders[i].bind("<Motion>",
                                       lambda event: self.update_entry_value(arr_of_sliders[i], arr_of_var[i]))
            return

        self.fig, self.ax1 = plt.subplots(1, 2 + (idx < 2), figsize=(18, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack()

        self.browse_button1 = tk.Button(self.frame, text=f"Browse Signal1)", command=self.browse_file1)
        self.browse_button1.pack()
        if idx < 2:
            self.browse_button2 = tk.Button(self.frame, text=f"Browse Signal2)", command=self.browse_file2)
            self.browse_button2.pack()

        # Constant value for multiply and shift
        if self.type in ['multiply', 'shift']:
            self.const_val_label = tk.Label(self.frame, text="Const value:")
            self.const_val_label.pack()
            self.const_val_entry = tk.Entry(self.frame)
            self.const_val_entry.pack()

        if self.type == 'normalize':
            self.min_val_label = tk.Label(self.frame, text="Min value:")
            self.min_val_label.pack()
            self.min_val_entry = tk.Entry(self.frame)
            self.min_val_entry.pack()

            self.max_val_label = tk.Label(self.frame, text="Max value:")
            self.max_val_label.pack()
            self.max_val_entry = tk.Entry(self.frame)
            self.max_val_entry.pack()

            self.a_b_choice = tk.StringVar()
            self.a_b_choice.set("option1")

            self.option1 = tk.Radiobutton(self.frame, text="-1 to 1", variable=self.a_b_choice, value="option1")
            self.option2 = tk.Radiobutton(self.frame, text="0 to 1", variable=self.a_b_choice, value="option2")

            self.option1.pack()
            self.option2.pack()

        self.display_button = tk.Button(self.frame, text=f"Display ({self.type})", command=self.display_plots)
        self.display_button.pack()

        self.data1 = None
        self.data2 = None

    def browse_file1(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            self.data1 = self.load_data(file_path)
            self.plot_data(self.ax1[0], self.data1, 'Signal1')

    def browse_file2(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            self.data2 = self.load_data(file_path)
            self.plot_data(self.ax1[1], self.data2, 'Signal2')

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = [list(map(float, line.split())) for line in lines[3:]]
        return data

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        self.file_path_var.set(file_path)

    def clear_ax(self, ax1):
        ax1.clear()
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Signal')

    def plot_loaded_signal(self):
        file_path = self.file_path_var.get()
        data = 0
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = [list(map(float, line.split())) for line in lines[3:]]
        self.plot_data(self.ax1, data, "Signal")
        self.canvas.draw()

    # Function to update entry value from slider
    def update_entry_value(self, slider, entry_var):
        entry_var.set(slider.get())

    def generate_signal_discrete(self, list, function_type):
        [A, theta, F, Fs, number_of_cycles] = list
        file_path = ""
        if function_type == 'Sine':
            file_path = "signal_discrete_sine.txt"
        else:
            file_path = "signal_discrete_cosine.txt"
        with open(file_path, 'w') as file:
            file.write("0\n0\n")
            file.write(f'{int(Fs)}\n')
            for i in range(int(Fs)):
                file.write(f'{i} ')
                if function_type == 'Sine':
                    file.write(f'{A * np.sin(2 * np.pi * F / Fs * i + theta)}\n')
                else:
                    file.write(f'{A * np.cos(2 * np.pi * F / Fs * i + theta)}\n')

    def generate_signal_continuous(self, list):
        file_path = "signal_continuous.txt"
        [A, theta, F, Fs, number_of_cycles] = list
        with open(file_path, 'w') as file:
            file.write("1\n1\n")
            file.write(f'{int(Fs)}\n')
            file.write(f'{F} {A} {theta}')

    def plot_type(self, ax, function_type, n, c, list):
        [A, theta, F, Fs, number_of_cycles] = list
        if function_type == 'Sine':
            signal_discrete = A * np.sin(2 * np.pi * F / Fs * n + theta)
        elif function_type == 'Cosine':
            signal_discrete = A * np.cos(2 * np.pi * F / Fs * n + theta)
        ax.plot(n / Fs, signal_discrete, color=c, label=function_type)
        ax.stem(n / Fs, signal_discrete, linefmt=c, markerfmt='o', basefmt=' ')
        self.generate_signal_discrete(list, function_type)
        self.generate_signal_continuous(list)

    # Function to plot the selected function
    def plot_function(self, function_choice, arr_of_sliders, ax1):
        function_type = function_choice.get()
        list = []
        for i in range(5):
            list.append(arr_of_sliders[i].get())
        [A, theta, F, Fs, number_of_cycles] = list
        print(list)
        if Fs < 2 * F:
            messagebox.showerror("Sampling Theorem Violation",
                                 "The sampling frequency (Fs) must be at least twice the analog frequency (F).")
            return
        n = np.arange(0, int(number_of_cycles * Fs / F))
        time = np.linspace(0, number_of_cycles / F)
        self.clear_ax(ax1)
        ax1.plot(time, np.zeros_like(time))
        if function_type == 'Both':
            self.plot_type(ax1, 'Sine', n, 'blue', list)
            self.plot_type(ax1, 'Cosine', n, 'orange', list)
        else:
            self.plot_type(ax1, function_type, n, 'blue', list)
        ax1.legend()
        self.canvas.draw()

    def plot_data(self, axis, data, label):
        if data:
            data = list(zip(*data))
            x, y = data[0], data[1]
            axis.clear()
            axis.plot(x, y, label=label)
            axis.set_xlabel('X')
            axis.set_ylabel('Y')
            axis.legend()

    def display_plots(self):
        if self.type == 'add':
            if self.data1 and self.data2:
                combined_data = [(x, y1 + y2) for (x, y1), (_, y2) in zip(self.data1, self.data2)]
                self.plot_data(self.ax1[2], combined_data, 'Signal1 + Signal2')
        elif self.type == 'sub':
            if self.data1 and self.data2:
                combined_data = [(x, y1 - y2) for (x, y1), (_, y2) in zip(self.data1, self.data2)]
                self.plot_data(self.ax1[2], combined_data, 'Signal1 - Signal2')
        elif self.type == 'multiply':
            if self.data1:
                val = int(self.const_val_entry.get())
                result = [(x, val * y) for (x, y) in self.data1]
                self.plot_data(self.ax1[1], result, f'{val} * Signal1')
        elif self.type == 'square':
            if self.data1:
                result = [(x, y**2) for (x, y) in self.data1]
                self.plot_data(self.ax1[1], result, 'Signal1 square')
        elif self.type == 'shift':
            if self.data1:
                val = int(self.const_val_entry.get())
                result = [(x + val, y) for (x, y) in self.data1]
                self.plot_data(self.ax1[1], result, f'Signal1 shifted by {val}')
        elif self.type == 'normalize':
            if self.data1:
                min_value = int(self.min_val_entry.get())
                max_value = int(self.max_val_entry.get())
                if self.a_b_choice.get() == 'option1':
                    a, b = -1, 1
                else:
                    a, b = 0, 1
                result = [(x, (y - min_value) / (max_value - min_value) * (b - a) + a) for (x, y) in self.data1]
                self.plot_data(self.ax1[1], result, 'Normalized signal')
        elif self.type == 'accumulate':
            if self.data1:
                acc_y = 0
                res = []
                for x, y in self.data1:
                    acc_y += y
                    res.append((x, acc_y))
                self.plot_data(self.ax1[1], res, 'Accumulation of Signal1')
        if self.data1:
            self.canvas.draw()
class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Plotter")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=1, fill='both')

        self.pages = []
        for i in range(8):
            page = PlotPage(self.notebook, i)
            self.pages.append(page)
        self.show_page(0)

    def show_page(self, page_num):
        self.notebook.select(page_num)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()