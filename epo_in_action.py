from tkinter import *
import tkinter.ttk
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
sns.set(rc={'figure.figsize':(20,10)})
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime

root = Tk()
root.title("DIY black-litterman: EPO for retail investors")
# root.iconbitmap("icon_1.ico")
#root.geometry("500x100")

title = Label(root, text='몇개의 종목에 투자하시겠습니까?')
title.grid(row=0,column=0,padx=10)

e=Entry(root, width =25, borderwidth=5)
e.grid(row=0,column=1)
e.insert(0, "")

def myClick0():

    global code1
    global code2
    global dct
    global dct1
    dct = dict()
    dct1 = dict()
    
    for i in range (1,int(e.get())+1) :    
        
        code1 = Entry(root, width =25, borderwidth=5)
        dct[i]=code1    
        code1.grid(row=i,column=1,padx=10)

    for i in range (1,int(e.get())+1) :    
        
        code2 = Entry(root, width =5, borderwidth=5)
        dct1[i]=code2    
        code2.grid(row=i,column=2,padx=10)

    for i in range (1,int(e.get())+1) :    
        title = Label(root,text=str(i)+"번")
        title.grid(row=i,column=0,padx=10)
    
    button0=Button(root,text='commit',command=myClick1,fg="blue",bg="white",padx=50)
    button0.grid(row=int(e.get())+1,column=0,columnspan=3,pady=10)
    
    
def myClick1():

    global code3
    global dct3
    dct3= dict()
    num=int(e.get())
    for i in range (num+3,num+5) :    
        code3 = Entry(root, width=25,borderwidth=5)
        dct3[i]=code3
        code3.grid(row=i,column=1,padx=5)
        
    dct3[num+3].insert(0,"0~1 값")
    dct3[num+4].insert(0,"YYYYMMDD")
    
    lbel = Label(root, text="확신수준")         
    lbel.grid(row=int(e.get())+3,column=0,padx=10)
    
    lbel= Label(root, text="시작일")         
    lbel.grid(row=int(e.get())+4,column=0,padx=10)
    
    button1=Button(root,text='Backtest Plot',command=backtest_plot, fg="blue",bg="white",padx=50)
    button1.grid(row=int(e.get())+6,column=0,pady=10)

    button2=Button(root,text='Performance',command=performance,fg="blue",bg="white",padx=50)
    button2.grid(row=int(e.get())+7,column=0,pady=10)

    button3=Button(root,text="EFficient Frontier", command=efficient_frontier,fg="blue",bg="white",padx=50)
    button3.grid(row=int(e.get())+6,column=1,pady=10)

    button4=Button(root,text="portfolio weights",command=weight_plot, fg="blue",bg='white',padx=50)
    button4.grid(row=int(e.get())+7,column=1,pady=10)
 
def performance():
    num=int(e.get())
    tickers=list(range(num))
    sig=list(range(num))
    for i in range(num):
        tickers[i]=dct[i+1].get()
        sig[i]=int(dct1[i+1].get())
    srk=(1-float(dct3[num+3].get()))
    time=dct3[num+4].get()
    rtn_panel = rtn_panel_maker(tickers,time,datetime.today())
    yours, zero_port, one_port = performance_measures(rtn_panel=rtn_panel, sig=sig,srk=srk)
    
    global new
    new = Toplevel()
    new.title("Performances")

    treeview = tkinter.ttk.Treeview(new,column=["kind","weights","Sharpe Ratio","MDD","Skew"],displaycolumns=["kind","weights","Sharpe Ratio","MDD","Skew"])
    treeview.pack()

    treeview.column("kind",width=50,anchor="center")
    treeview.heading("kind",text="SRK",anchor="center")

    treeview.column("weights",width=200,anchor="center")
    treeview.heading("weights",text="Weights",anchor="center")

    treeview.column("Sharpe Ratio",width=50,anchor="center")
    treeview.heading("Sharpe Ratio",text="SR",anchor="center")

    treeview.column("MDD",width=50,anchor="center")
    treeview.heading("MDD",text="MDD",anchor="center")

    treeview.column("Skew",width=50,anchor="center")
    treeview.heading("Skew",text="Skew",anchor="center")

    treeview['show'] = "headings"

    treevaluelist = [(str(srk),yours['weights'],yours['SR'],yours['MDD'],yours['SKEW']),
                    (str(0),zero_port['weights'],zero_port['SR'],zero_port['MDD'],zero_port['SKEW']),
                    (str(1),one_port['weights'],one_port['SR'],one_port['MDD'],one_port['SKEW'])]

    for i in range(len(treevaluelist)):
        treeview.insert("","end",text="",values=treevaluelist[i],iid=i)

    new.mainloop()

def efficient_frontier():
    num=int(e.get())
    tickers=list(range(num))
    sig=list(range(num))
    for i in range(num):
        tickers[i]=dct[i+1].get()
        sig[i]=int(dct1[i+1].get())
    srk=(1-float(dct3[num+3].get()))
    time=dct3[num+4].get()

    axx = plotting_ef(num=num,tickers = tickers, sig = sig, srk = srk, time = time)

    root2=Tk()
    root2.title("Efficient Frontier")
    canvas = FigureCanvasTkAgg(axx, master=root2) 
    canvas.get_tk_widget().pack()

    root2.mainloop()


def backtest_plot():
    num=int(e.get())
    tickers=list(range(num))
    sig=list(range(num))
    for i in range(num):
        tickers[i]=dct[i+1].get()
        sig[i]=int(dct1[i+1].get())
    srk=(1-float(dct3[num+3].get()))
    time=dct3[num+4].get()
    rtn_panel = rtn_panel_maker(tickers,time,datetime.today())

    axx = chart(rtn_panel,sig,srk)
    
    root2=Tk()
    root2.title("backtest_plot")
    canvas = FigureCanvasTkAgg(axx, master=root2) 
    canvas.get_tk_widget().pack()

    root2.mainloop()

def weight_plot():
    num=int(e.get())
    tickers=list(range(num))
    sig=list(range(num))
    for i in range(num):
        tickers[i]=dct[i+1].get()
        sig[i]=int(dct1[i+1].get())
    srk=(1-float(dct3[num+3].get()))
    time=dct3[num+4].get()

    axx= plotting_pie(tickers=tickers,time=time,sig=sig,shrinkage=srk)

    root2=Tk()
    root2.title("weight plot")
    canvas = FigureCanvasTkAgg(axx,master=root2)
    canvas.get_tk_widget().pack()
    
    root2.mainloop()


mybutton = Button(root,text='commit',command=myClick0,fg="blue",bg="white",padx=10)
mybutton.grid(row=0,column=2,pady=10,padx=8)


root.mainloop()