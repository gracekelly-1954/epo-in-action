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

def get_mdd(x):
    arr_v = np.array(x)
    peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
    peak_upper = np.argmax(arr_v[:peak_lower])
    return (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]

def anchored_epo(rtn_panel, signal, shrinkage):
    s = np.array(signal) / np.array(signal).sum() 
    n = len(rtn_panel.columns)
    shrinkage = shrinkage
    vcov = rtn_panel.cov()
    corr = rtn_panel.corr()
    v = np.zeros((n,n))
    np.fill_diagonal(v,np.diag(vcov))
    std = np.sqrt(v)
    d = np.diag(vcov)
    a = (1/d) / (1/d).sum()
    shrunk_corr = (((1-shrinkage) * np.identity(n)) @ corr) + (shrinkage * np.identity(n))
    shrunk_vcov = std @ ( (((1-shrinkage) * np.identity(n)) @ shrunk_corr) + (shrinkage * np.identity(n)) ) @ std
    better_risk_estimate = std @ shrunk_corr @ std
    inverse_shrunk_vcov = np.linalg.inv(shrunk_vcov)
    anchored_epo = inverse_shrunk_vcov @ (((1-shrinkage) * (np.sqrt(a.T@better_risk_estimate@a) / np.sqrt(s.T@inverse_shrunk_vcov@better_risk_estimate@inverse_shrunk_vcov@s)) * s) +((shrinkage*np.identity(n))@v@a))
    for i in range(len(anchored_epo)):
        if anchored_epo[i] < 0:
            anchored_epo[i] = 0
        else:
            pass    
    anchored_epo = anchored_epo * (1/anchored_epo.sum())
    return anchored_epo

def rtn_panel_maker(tickers, start, end):
    tickers = tickers
    start = start[:4] +'-'+ start[4:6] +'-' + start[6:]
    end = end
    
    rtn_panel = pd.DataFrame()

    for ticker in tickers:
        rtn_panel[ticker] =yf.download(ticker,start = start)['Adj Close'].pct_change()
        
    return rtn_panel

def chart(rtn_panel,sig,srk):
    acc=(1+(rtn_panel @ anchored_epo(rtn_panel = rtn_panel, signal = sig, shrinkage = srk).T)).cumprod()
    acc0=(1+(rtn_panel @ anchored_epo(rtn_panel = rtn_panel, signal = sig, shrinkage = 0).T)).cumprod()
    acc1=(1+(rtn_panel @ anchored_epo(rtn_panel = rtn_panel, signal = sig, shrinkage = 1).T)).cumprod()
    accT=pd.concat([acc,acc0,acc1],axis=1)
    accT.columns=([srk,'0','1'])
    fig=Figure(figsize=(8,5),dpi=100)
    fig.add_subplot(1,1,1).plot(accT)
    fig.legend([srk,'0','1'],bbox_to_anchor=(0.24, 0.86))
    
    return fig

def performance_measures(rtn_panel, sig, srk):
    acc=(rtn_panel @ anchored_epo(rtn_panel = rtn_panel, signal = sig, shrinkage = srk).T)
    acc0=(rtn_panel @ anchored_epo(rtn_panel = rtn_panel, signal = sig, shrinkage = 0).T)
    acc1=(rtn_panel @ anchored_epo(rtn_panel = rtn_panel, signal = sig, shrinkage = 1).T)
    accT=pd.concat([acc,acc0,acc1],axis=1)
    accT.columns=(['yours','0','1'])
    
    yours = dict()
    zero_port = dict()
    one_port = dict()

    yours['weights'] = np.round(list(anchored_epo(rtn_panel = rtn_panel, signal=sig, shrinkage = srk)),2)
    zero_port['weights'] = np.round(list(anchored_epo(rtn_panel = rtn_panel, signal=sig, shrinkage = 0)),2)
    one_port['weights'] = np.round(list(anchored_epo(rtn_panel = rtn_panel, signal=sig, shrinkage = 1)),2)

    yours['SR'] = np.round(( accT['yours'].mean() / accT['yours'].std() ) * np.sqrt(252) ,2)
    zero_port['SR'] = np.round(( accT['0'].mean() / accT['0'].std() ) * np.sqrt(252) ,2)
    one_port['SR'] = np.round(( accT['1'].mean() / accT['1'].std() ) * np.sqrt(252) ,2)

    yours['MDD'] = np.round(get_mdd((1+accT['yours'].iloc[1:]).cumprod()) * 100,2)
    zero_port['MDD'] = np.round(get_mdd((1+accT['0'].iloc[1:]).cumprod()) * 100,2)
    one_port['MDD'] = np.round(get_mdd((1+accT['1'].iloc[1:]).cumprod()) * 100,2)

    yours['SKEW'] = np.round(accT['yours'].skew(),2)
    zero_port['SKEW'] = np.round(accT['0'].skew(),2)
    one_port['SKEW'] =  np.round(accT['1'].skew(),2)

    return yours, zero_port, one_port

def plotting_ef(num,tickers,sig,srk,time):
    rtn_panel = rtn_panel_maker(tickers=tickers,start=time,end=datetime.today()).iloc[1:]
    np.random.seed(42)
    num_ports = 3000
    all_weights = np.zeros((num_ports, num))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for x in range(num_ports):
        # Weights
        weights = np.array(np.random.random(num))
        weights = weights/np.sum(weights)
        
        # Save weights
        all_weights[x,:] = weights
        
        # Expected return
        ret_arr[x] = np.sum( (rtn_panel.mean() * weights * 252))
        
        # Expected volatility
        vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(rtn_panel.cov()*252, weights)))
        
        # Sharpe Ratio
        sharpe_arr[x] = ret_arr[x]/vol_arr[x]

    acc=(rtn_panel @ anchored_epo(rtn_panel = rtn_panel, signal = sig, shrinkage = srk).T)
    acc0=(rtn_panel @ anchored_epo(rtn_panel = rtn_panel, signal = sig, shrinkage = 0).T)
    acc1=(rtn_panel @ anchored_epo(rtn_panel = rtn_panel, signal = sig, shrinkage = 1).T)
    accy = acc.mean()*252
    acc0y = acc0.mean()*252
    acc1y = acc1.mean()*252

    figure = plt.figure(figsize=(12,8))
    plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
    plt.grid(True)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.scatter(acc.std()*np.sqrt(252), accy,c='red', s=50,marker='x')
    plt.scatter(acc0.std()*np.sqrt(252), acc0y,c='red',s=50,marker='o')
    plt.scatter(acc1.std()*np.sqrt(252),acc1y,c='red',s=50,marker='+')

    return figure

def plotting_pie(tickers,time,sig,shrinkage):
    rtn_panel = rtn_panel_maker(tickers,time,datetime.today()).iloc[1:]
    yours, zero_port, one_port = anchored_epo(rtn_panel=rtn_panel, signal=sig, shrinkage=shrinkage), anchored_epo(rtn_panel=rtn_panel,signal=sig,shrinkage=0),anchored_epo(rtn_panel=rtn_panel,signal=sig,shrinkage=1)
    ratio_yours = yours
    ratio_zero = zero_port
    ratio_one = one_port
    labels=tickers
    fig = plt.figure()
    fig1 = fig.add_subplot(2,2,1)
    fig2 = fig.add_subplot(2,2,3)
    fig3 = fig.add_subplot(2,2,4)
    fig1.pie(ratio_yours,labels=labels,autopct='%.1f%%',startangle=270,counterclock=False)
    fig1.set_title(str(shrinkage))
    fig2.pie(ratio_zero,labels=labels,autopct='%.1f%%',startangle=270,counterclock=False)
    fig2.set_title(str(0))
    fig3.pie(ratio_one,labels=labels,autopct='%.1f%%',startangle=270,counterclock=False)
    fig3.set_title(str(1))
    
    return fig
    
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