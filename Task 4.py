import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os, math, cmath

amp_list = []
theta_list = []
samples_list = []
input_fs =4


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

def convert_exp_to_complex(comp):
    # +ve
    if comp == abs(comp):
        x = math.cos(comp.imag) + 1j * math.sin(comp.imag)
    else:
        x = math.cos(comp.imag) - 1j * math.sin(comp.imag)
    # print(f"x = {x}")
    return complex(round(x.real, 7), round(x.imag, 7))

#Use to test the Amplitude of DFT and IDFT
def SignalComapreAmplitude(SignalInput = [] ,SignalOutput= []):
    if len(SignalInput) != len(SignalInput):
        return False
    else:
        for i in range(len(SignalInput)):
            if abs(SignalInput[i]-SignalOutput[i])>0.001:
                return False
        return True

def RoundPhaseShift(P):
    while P<0:
        P+=2*math.pi
    return float(P%(2*math.pi))

#Use to test the PhaseShift of DFT
def SignalComaprePhaseShift(SignalInput = [] ,SignalOutput= []):
    if len(SignalInput) != len(SignalInput):
        return False
    else:
        for i in range(len(SignalInput)):
            A=RoundPhaseShift(SignalInput[i])
            B=RoundPhaseShift(SignalOutput[i])
            if abs(A-B)>0.0001:
                return False
        return True
def update_entry_value(slider, entry_var):
    entry_var.set(slider.get())


def generate_file(name,is_freq, n, x, y):
    with open(f"{name}.txt", 'w') as file:
        file.write(f'0\n{is_freq}\n{n}\n')
        for i in range(int(n)):
            file.write(f'{x[i]} {y[i]}\n')
def IDFT(amp_l,theta_l):
    samples=[]
    N = len(amp_list)
    x=[]
    for i in range(N):
        x.append(cmath.rect(amp_l[i],theta_l[i]))
    for n in range(N):
        sum = 0 + 0j
        for k in range(N):
            theta = 2 * np.pi * k * n / N
            sum += x[k] * (math.cos(theta) + 1j * math.sin(theta))
        sum /= N
        samples.append(round(sum.real))
    return samples

def DFT(samples_list):
    amp_l = []
    theta_l=[]
    N = len(samples_list)
    for k in range(N):
        sum = 0+0j
        for n in range(N):
            theta = 2 * np.pi * k * n / N
            sum += samples_list[n] * (math.cos(theta) - 1j * math.sin(theta))
        r, th = cmath.polar(sum)
        amp_l.append(r)
        theta_l.append(th)
    return amp_l,theta_l

def DCT(sample_l):
    dct_list = []
    N = len(sample_l)
    constant = math.sqrt(2/N)
    pi_over_4N = (np.pi/(4*N))
    for k in range(N):
        sum = 0
        for n in range(N):
            sum += sample_l[n]*math.cos(pi_over_4N*(2*n-1)*(2*k-1))
        dct_list.append(constant*sum)
    return dct_list


def compare(DFT):
    Expected_amp_list =[]
    Expected_theta_list=[]
    file_path= ""
    if DFT:
        file_path = "Output_Signal_DFT_A,Phase.txt"
    else:
        file_path=  "Output_Signal_IDFT.txt"
    with open(file_path, 'r') as file:
        lines = file.readlines()[3:]
        for line in lines:
            r, th = 0, 0
            # print(line)
            parts = list(line.strip().split(' '))
            if 'f' in parts[0]:
                r = float(parts[0][:-1])
            else:
                r = float(parts[0])
            if 'f' in parts[1]:
                th = float(parts[1][:-1])
            else:
                th = float(parts[1])
            if DFT:
                Expected_amp_list.append(r)
                Expected_theta_list.append(th)
            else:
                Expected_amp_list.append(th)
    if not DFT:
        if SignalComapreAmplitude(samples_list,Expected_amp_list):
            messagebox.showinfo("TEST","Accepted")
            return
    else:
        if SignalComapreAmplitude(amp_list,Expected_amp_list) and SignalComaprePhaseShift(theta_list,Expected_theta_list):
            messagebox.showinfo("TEST", "Accepted")
            return
    messagebox.showinfo("TEST","Wrong Answer!!")

def Plot_Signals(amp_l,sample_l,fs,ax1,ax2,c1,c2,name):
    x = []
    n = len(sample_l)
    x.append(2*np.pi/n*fs)
    for i in range(1,n):
        x.append(x[0]+x[i-1])
    ax1.clear()
    ax1.stem(x, amp_l)
    ax1.set_xlabel('Frequency (HZ)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Frequency domain of '+name)
    ax2.clear()
    ax2.plot(np.arange(n), sample_l)
    ax2.set_xlabel('Index(n)')
    ax2.set_ylabel('Amplitude - x(n) ')
    ax2.set_title('Signal of '+name)
    c1.draw()
    c2.draw()
def browse_file():
    file_path = filedialog.askopenfilename()
    file_path_var.set(file_path)
    file_path = file_path_var.get()

    try:
        global amp_list
        global theta_list
        global samples_list
        amp_list.clear()
        theta_list.clear()
        samples_list.clear()
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Always = 0
            is_periodic = int(lines[0].strip())
            signal_type = int(lines[1].strip())
            data = lines[3:]
            if signal_type == 0:  # Time
                for line in data:
                    parts = list(line.strip().split(' '))
                    samples_list.append(float(parts[1]))
                amp_list,theta_list=DFT(samples_list)
            else:
                for line in data:
                    r,th=0,0
                    parts = list(line.strip().split(' '))
                    if 'f' in parts[0]:
                        r = float(parts[0][:-1])
                    else:
                        r = float(parts[0])
                    if 'f' in parts[1]:
                        th = float(parts[1][:-1])
                    else:
                        th = float(parts[1])
                    amp_list.append(r)
                    theta_list.append(th)
                samples_list = IDFT(amp_list,theta_list)
            Plot_Signals(amp_list,samples_list,input_fs,ax1,ax3,canvas,canvas3,'input')
    except Exception as e:
        print("ERROR HERE!")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def Edit_signal(edit_input):
    try:
       listy = []
       print("Edit signal")
       for i in range(4):
           listy.append(arr_of_sliders[i].get())
       [amp, theta, idx ,fs] = listy
       idx = int(idx)
       tmp_amp,tmp_theta = amp_list[idx],theta_list[idx]
       amp_list[idx]=amp
       theta_list[idx]=theta
       new_samples = IDFT(amp_list,theta_list)
       print("Samples After last modifications: ", new_samples)
       Plot_Signals(amp_list,new_samples,fs,ax2,ax4,canvas2,canvas4,'output')
       if not edit_input:
           amp_list[idx] ,theta_list[idx] = tmp_amp,tmp_theta
       else:
           samples_list = new_samples
           input_fs = fs
           Plot_Signals(amp_list, samples_list, input_fs, ax1, ax3, canvas, canvas3,'input')
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def plot_dct_signal():
    global samples_list
    new_amp=DCT(samples_list)
    new_theta = []
    for i in range(len(samples_list)):
        new_theta.append(0)
    new_samples=IDFT(new_amp,new_theta)
    Plot_Signals(new_amp,new_samples,input_fs,ax2,ax4,canvas2,canvas4,'output')
    SignalSamplesAreEqual("DCT\DCT_output.txt",new_amp)
    m = arr_of_sliders[4].get()
    generate_file("DCT_freq",1,m,new_theta,new_amp)
    generate_file("DCT_time",0,m,np.arange(m),new_samples)
def edit_input():
    Edit_signal(True)
def edit_output():
    Edit_signal(False)
def remove_dc_component():
    # Calculate the mean
    mean = sum(samples_list) / len(samples_list)
    new_samples = [x - mean for x in samples_list]
    new_amp,new_theta = DFT(new_samples)
    Plot_Signals(new_amp,new_samples,input_fs,ax2,ax4,canvas2,canvas4,'output')
    SignalSamplesAreEqual("Remove DC component/DC_component_output.txt",[round(x,3)for x in new_samples])
    print(new_samples)


app = tk.Tk()
app.title("DSP")

# Set the default window size to 1080p resolution
app_width = 1920
app_height = 1080
app.geometry(f"{app_width}x{app_height}")

# Variables for input fields
arr_of_var = []
for i in range(5):
    x = tk.StringVar()
    arr_of_var.append(x)


# Create a tkinter frame
frame = ttk.Frame(app)
frame.pack(expand=True, fill='both', padx=10, pady=10)

# Create a big title label at the top
big_title_label = ttk.Label(frame, text="Function Plotter", font=("Arial", 24))
big_title_label.grid(row=0, column=0, columnspan=3, padx=5, pady=20, sticky='n')

# Create a frame to center the elements in the middle
center_frame = ttk.Frame(frame)
center_frame.grid(row=1, column=0, columnspan=3)



arr_of_text = ["Amplitude (A):", "Phase Shift (radians):", "index of signal (zero based):","Fs","m(save m when computing DCT)"]
arr_of_sliders = []

for i in range(5):
    label = ttk.Label(center_frame, text=arr_of_text[i])
    label.grid(row=i + 1, column=0, padx=5, pady=5)
    slider = ttk.Scale(center_frame, from_=0, to=10, orient='horizontal', length=200, variable=arr_of_var[i])
    slider.grid(row=i + 1, column=1, padx=5, pady=5)
    entry = ttk.Entry(center_frame, textvariable=arr_of_var[i])
    entry.grid(row=i + 1, column=2, padx=5, pady=5)
    arr_of_var[i].set(slider.get())
    arr_of_sliders.append(slider)

# Entry and Browse button for file path
file_path_var = tk.StringVar()
file_path_entry = ttk.Entry(center_frame, textvariable=file_path_var, width=40)
file_path_entry.grid(row=8, column=0, padx=5, pady=5)

browse_button = ttk.Button(center_frame, text="Browse", command=browse_file)
browse_button.grid(row=8, column=1, padx=5, pady=5)

# Load button for loaded signal
edit_output_button = ttk.Button(center_frame, text="Show the Signal after modification", command=edit_output)
edit_output_button.grid(row=8, column=2, columnspan=1, padx=5, pady=10)


edit_input_button = ttk.Button(center_frame, text="Edit in input signal", command=edit_input)
edit_input_button.grid(row=8, column=3, columnspan=1, padx=5, pady=10)

plot_dct = ttk.Button(center_frame, text="Show DCT of input signal", command=plot_dct_signal)
plot_dct.grid(row=8, column=5, columnspan=1, padx=5, pady=10)

remove_dc = ttk.Button(center_frame, text="remove DC component", command=remove_dc_component)
remove_dc.grid(row=8, column=6, columnspan=1, padx=5, pady=10)

# Create a Matplotlib figure and canvas for DFT plot
fig, ax1 = plt.subplots(figsize=(app_width / 200, app_height / 300), dpi=96)
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=2, column=0, padx=5, pady=5)

# Create a Matplotlib figure and canvas for Phase plot
fig2, ax2 = plt.subplots(figsize=(app_width / 200, app_height / 300), dpi=96)
canvas2 = FigureCanvasTkAgg(fig2, master=frame)
canvas_widget2 = canvas2.get_tk_widget()
canvas_widget2.grid(row=2, column=1, padx=5, pady=5)

# Create a Matplotlib figure and canvas for Phase plot
fig3, ax3 = plt.subplots(figsize=(app_width / 200, app_height / 300), dpi=98)
canvas3 = FigureCanvasTkAgg(fig3, master=frame)
canvas_widget3 = canvas3.get_tk_widget()
canvas_widget3.grid(row=3, column=0, padx=5, pady=5)

# Create a Matplotlib figure and canvas for New signal  plot
fig4, ax4 = plt.subplots(figsize=(app_width / 200, app_height / 300), dpi=98)
canvas4 = FigureCanvasTkAgg(fig4, master=frame)
canvas_widget4 = canvas4.get_tk_widget()
canvas_widget4.grid(row=3, column=1, padx=5, pady=5)

for i in range(5):
    arr_of_sliders[i].bind("<Motion>", lambda event: update_entry_value(arr_of_sliders[i], arr_of_var[i]))

app.mainloop()
