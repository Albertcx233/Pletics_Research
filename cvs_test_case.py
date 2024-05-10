from ast import Dict
from typing import OrderedDict
import pandas as pd
#import matplotlib
#matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
#print("Switched to:",matplotlib.get_backend())
import scipy
import numpy
import math
import os
from datetime import datetime
from matplotlib.widgets import CheckButtons





'''Namespace'''
MAKE_AND_MODEL = {"canon":['mb2120', 'mf751cdw'], "brother":['hl-l3290', 'mfc-j4535dw'], "hp":['mfpcdw','mfpfdw','9015e','7955e'], "epson":['wf4830']}

TEST_MODES = ['direct', 'router', 'router-ethernet']
#UNIT_MEASURED = ['printer', 'router-and-printer']
#added ''router-ethernet' for our wired connection things

UNIT_MEASURED = ['p', 'rp']

#having router and router_printer becomes "router_router_printer" vs "router_printer" should we change this? instead we should use "router-and-printer"
# or the simpler "p" and "rp"



STEPS = ['off', '5', '6','7','8','9','10']

# 6 7 8 9 are jobs 1 2 3 4 respectively
# 1 is off
# 5 is 1 hour of sleep
# 10 is the multiple hours auto-off step that also can include the off step, we changed our method part way through but having the option for all the things we did is important
"""
[make]_[model]_[test_mode]_[unit_measured]_[step].hobo
"""

def csv_fix(listcsv:list):
    #cwd = os.getcwd()
    new_list=[]
    for currentcsv in listcsv:
        #currentpath = cwd + '/csvsfromhoboware/' + currentcsv
        currentpath=currentcsv
        ourtable = pd.read_csv(currentpath, skiprows = [0])

        labels = ['num', 'date_time', 'RMS_V', 'RMS_A', 'W', 'Wh', 'VA',
              'PF']  # new labels to replace the old bad ones with the serial numbers

        tableshape = ourtable.shape

        size = ourtable.columns.size

        x = 0

        while x < 8:
            oldlabel = ourtable.columns[x]
            newlabel = labels[x]

            ourtable.rename(columns = {oldlabel: newlabel}, inplace = True)
            x += 1

        ourtable.drop(labels = 'num', axis = 1, inplace = True)
        # print(str(ourtable.iloc[-1,4]))

        if str(ourtable.iloc[
               -1, 4]) == "nan":  # some of the datasets have the last row just showing that the recording was stopped, this is unneccesary because we can see that it ends because its the end of the file
            ourtable.drop(ourtable.tail(1).index, inplace = True)
        # print("dropping last row")
    # print(str(ourtable.iloc[-1,4]))

        reducedtable = ourtable.filter(['date_time', 'RMS_V', 'RMS_A', 'W', 'Wh', 'VA', 'PF'])
        new_list.append(reducedtable)
    return new_list



def wdiff(df):
    wfloatingsum = 0
    wfloatingavg = 0
    wpreviousavg = 0
    wlastfew = []  #list used to calculate the floating average
    wdifferent_bools = []  # list that will become a new column in the dataframe
    wdiff_binary = []
    x = 0
    lentable = len(df)  # list that will become a new column in the dataframe
    wdiff_binary = []
    for x in range(len(df)):  # somewhat out dated, will be superceded by looking at the derivative of the log of y. THIS HAS BEEN DONE
        wcurrent = df.iloc[x, 3]
        #print(wcurrent)

        if x > 0:
            wpreviousavg = wfloatingavg

        wlastfew.append(df.iloc[x, 3])

        if len(wlastfew) > 3:  # this adjusts the length of the floating average
            wlastfew.pop(0)

        wfloatingavg = sum(wlastfew) / len(wlastfew)

        isdifferent_bool = False
        if x > 0:
            pct_change = 0
            if wcurrent > 0.005:
                pct_change = (wcurrent - wpreviousavg) / wcurrent
            else:
                pct_change = wcurrent - wpreviousavg

            if abs(pct_change) > .15:  # percent change sensitivity
                if (abs(wcurrent - wpreviousavg) > .5):  # flat change sensitivity
                    isdifferent_bool = True

        wdifferent_bools.append(isdifferent_bool)


        wavgs = [] #list that houses the floating average of x-1 x x+1
        wmidavgtable = []
        for x in range(lentable): #this one is getting the x-1 x x+1 values
            if x == 0:
                wmidavgtable = [df.iloc[x,3], df.iloc[x+1,3]]
            elif x == lentable - 1:
                wmidavgtable = [df.iloc[x,3], df.iloc[x-1,3]]
            else:
                wmidavgtable = [df.iloc[x-1,3], df.iloc[x,3], df.iloc[x+1,3]]
            wmidavg = sum(wmidavgtable)/len(wmidavgtable)
            wavgs.append(wmidavg)



        #####
        #print(len(df))
        #print(len(wdifferent_bools))
        print(f'{int(len(wdifferent_bools)/len(df)*100)}%')

    df['nn_Avg'] = wavgs
    df['is_diff'] = wdifferent_bools
    df["date_time"]= pd.to_datetime(df['date_time'], format="%m/%d/%y %I:%M:%S %p")
    df['state'] = numpy.nan
    return df




###state
#      0           1       2      3    4     5     6      7          8         9
# ['date_time', 'RMS_V', 'RMS_A', 'W', 'Wh', 'VA', 'PF', 'Is_diff', 'nn_Avg', 'state]



######geting state label



























#testcase1=csv_fix(['hp_7995e_direct_p_9.csv',"brother_hl-l3290_direct_p_6.csv"])
#print(testcase1)
#testcase2=wdiff(testcase1[0])


print("1")
def new_df(df):

    ######geting state label
    yw =df.W
    ###############################################################################################################################################
    xax = df.date_time
    xstart = xax.iloc[0]
    deltax = []
    xaxdt = []
    ysqlog = []
    for x in range(len(yw)):
        # ysqlog.append( math.sqrt(yw.iloc[x]))
        ysqlog.append(math.log(yw.iloc[x] + 1))

    yavg = df.nn_Avg
    # getting the derivative of the nn_average
    ydiff = numpy.diff(yw)

    ysqlogdiff = numpy.diff(ysqlog)

    xdtdiff = []
    for n in range(len(xaxdt) - 1):
        xdtdiff.append(xaxdt[n])
    z = df.is_diff.astype(float)
    y_sg = scipy.signal.savgol_filter(yw, 4, 2)
    ydiff_sg = scipy.signal.savgol_filter(ydiff, 30, 4)
    ysqlog_sg = scipy.signal.savgol_filter(ysqlogdiff, 4, 2)

    is_stable = []

    numstates = 0;
    laststab = True

    if ysqlogdiff[0] < 0.5:
        laststab = True
    else:
        laststab = False

    numstablestates = 0
    numtransitions = 0

    for f_iter in range(
            len(xax)):  # this is where we go through the diff to see when its low, because when it is near 0 that is a stable state

        if f_iter < len(yw) - 1:

            if (numpy.absolute(ysqlogdiff[f_iter]) < 0.30):  # the sensitivity to the state changes
                is_stable.append(True)
                # if laststab != True: #thought i needed this logic, but oyu can just count them in the next state
                #    numstates+=1
                #    numstablestates +=1
                laststab = True

            else:
                is_stable.append(False)
                # if laststab != False:
                #    numstates +=1
                #    numtransitions +=1
                laststab = False
        else:
            is_stable.append(laststab)
    # print(numstablestates)

    currentstate = []
    listofstates = []
    if ysqlogdiff[0] < 0.5:
        laststab = True
    else:
        laststab = False

    for f_iter in range(
            len(is_stable)):  # looks for states (range of data points) where the power consumption is stable as marked in the previous for statement and groups them
        if is_stable[f_iter] == True:
            currentstate.append(df.iloc[f_iter, :])
            laststab = True
        else:
            if laststab == True:  # if that last state was stable, and it is no longer stable, appened to the this state to the list of states
                listofstates.append(currentstate)
                currentstate = []
            laststab = False
    if len(currentstate) >= 1:  # if the loop ends with a stable state, append it to the list of states
        listofstates.append(currentstate)

    listofstatesavg = []  # we need to get the states and, if there are any that are really close to each other merge them, then rank them by order of lowest energy consumption to highest
    dictofstates = {}

    for f_iter in range(len(listofstates)):
        currentstate = pd.DataFrame(listofstates[f_iter])
        c_avg_state = currentstate.loc[:, 'W'].mean()
        listofstatesavg.append(c_avg_state)

        if c_avg_state in dictofstates:  # dicts cannot have duplicates so on the offchance that 2 sections have the same average we need to account for that
            dictofstates[c_avg_state].append(currentstate)
        else:
            dictofstates[c_avg_state] = currentstate

    sortedstates = OrderedDict(sorted(
        dictofstates.items()))  # this creates a new dict from the old one that is ordered based on the labels, which is the average


    state_label = 0
    currentaverage = 0
    lastaverage = -1  # negative means it hasn't been set yet
    # print(sortedstates)


    for currentaverage in sortedstates:  # we now have the states sorted in ascending order, we need to label reducedtable with our corresponding

        if lastaverage >= 0:  # see if the last state is significantly far enough from the last one. But can't do that on the first iteration, so last average needs to be set
            average_diff = abs(currentaverage - lastaverage)
            if average_diff > 0.1 and state_label < 6:
                state_label += 2
        lastaverage = currentaverage  # sets last average for the next loop
        currentstate = sortedstates[currentaverage]
        # print(str(currentaverage) + ' ' + str(state_label))
        # print(currentstate)

        for f_iter in range(len(currentstate)):
            currenttime = currentstate.iloc[f_iter, 0]

            time_index = df.loc[(df == currenttime).any(axis = 1)].index[0]

            df.iloc[time_index, 9] = state_label
    # the states we be represented as
    # 0 off
    # 1 from off <-> standby
    # 2 standby/lpm
    # 3 standby <-> ready
    # 4 ready/idle
    # 5 ready <-> active
    # 6 active/printing


    df['state']=df["state"].map({0:"off",2:"standby/lpm",4:"ready/idle",6:"active/printing"})


    return df





import matplotlib.pyplot as plt


if __name__ =="__main__":
    print("1")
    file_list=['hp_7995e_direct_p_9.csv',"hp_7995e_direct_p_8.csv","hp_7995e_direct_p_7.csv"]
    df_list=csv_fix(['hp_7995e_direct_p_9.csv',"hp_7995e_direct_p_8.csv","hp_7995e_direct_p_7.csv"])
    print(">>>>>")
    i=0
    for df in df_list:
        print(">>>>>>>>>>>>>>>")
        state_info_dict={}
        df=wdiff(df)
        df=new_df(df)
        state_info_dict=df["state"].value_counts()
        state_info_dict=state_info_dict.to_dict()

        bars=plt.bar(range(len(state_info_dict)),list(state_info_dict.values()),align="center")
        plt.xticks(range(len(state_info_dict)),list(state_info_dict.keys()))
        plt.xlabel("State")
        plt.ylabel("Total time by second")
        plt.title(f"{file_list[i]}")

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha = 'center',
                     va = 'bottom')
        plt.show()
        plt.savefig(f"{file_list[i]}:Total Time per state.png")
        i+=0
    i=0
    for df in df_list:
        power_dict={}
        state_type=list(df["state"].unique())
        for type in state_type:
            df_temp=df[df["state"]==type]
            avg_pf=df_temp["PF"].mean()
            power_dict[type]=avg_pf

        bars=plt.bar(range(len(power_dict)),list(power_dict.values()),align="center")
        plt.xticks(range(len(power_dict)),list(power_dict.keys()))
        plt.xlabel("State")
        plt.xlabel("Avg_PF")
        plt.title(f"{file_list[i]}")
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha = 'center',
                     va = 'bottom')
        plt.show()
        plt.savefig(f"{file_list[i]}:Avg_PF per state .png")
        i+=1







