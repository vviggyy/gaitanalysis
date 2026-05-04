## All Functions for Step Frequency analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import numpy.linalg as LA
from scipy.signal import find_peaks
from scipy.stats import linregress

## Regressionsgerade Gangrichtung
## Dient als Grundlage für das korrigieren der Linie auf slope 0
def plt_regression(lane, com_data, lf_data, rf_data, plot='off'):
    Testline_Lane = []
    Testline_Lane = np.array([com_data[lane][0][1,:]] + [com_data[lane][1][1,:]])
    Line_Lane = np.array([Testline_Lane[:,0]] + [Testline_Lane[:,-1]])
    LL = abs(Line_Lane[1,:] - Line_Lane[0,:])
    #slope, intercept, r_value, p_value, std_err = linregress((Line_Lane[:,0], Line_Lane[:,1]))
    slope, intercept, r_value, p_value, std_err = linregress(Line_Lane[:,0], Line_Lane[:,1])
    
    if plot == 'show':
        print("---- Regression line ----")
        plt.axis('equal')
        plt.plot(com_data[lane][0][1,:], com_data[lane][1][1,:])
        plt.axis('equal')
        plt.plot(Line_Lane[:,0], Line_Lane[:,1])
        plt.plot(com_data[lane][0][1,:], com_data[lane][1][1,:])
        plt.show()
        return LL, slope
    elif plot == 'off':
        return LL, slope
    else:
        print(plot + " not defined! Use 'off' or 'show'.")
        return KeyError

def get_gait(lane, com_data, lf_data, rf_data, plot='off'):
    LL, slope = plt_regression(lane, com_data, lf_data, rf_data)
    # um wie viel Grad ist die Ganggerade gedreht
    a = np.array([1, 0])
    b = copy.copy(LL)
    inner = np.inner(a, b)
    norms = LA.norm(a) * LA.norm(b)
    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg = np.rad2deg(rad)
    #print(f'Deg: {deg}')
    # Rotationsmatrix
    # https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))
    #print(rotation_matrix)
    rot_a = c
    rot_b = -s
    rot_c = s
    rot_d = c
    # Vektor mit x und y (allerings Achsen andersrum, damit es visuell besser
    # aussieht) erstellen
    
    fig, ax = plt.subplots(2,3)
    COM_xy_Lane = np.array([com_data[lane][1][1,:]] + [com_data[lane][0][1,:]])
    COM_xy_Lanetp = COM_xy_Lane.transpose()
    plt.axis('equal')
    ax[0][0].plot(COM_xy_Lanetp[:,0], COM_xy_Lanetp[:,1])
    ax[0][0].set_title("COM_xy_pretrans")
    
    LF_xy_Lane = np.array([lf_data[lane][1]] + [lf_data[lane][0]])
    LF_xy_Lanetp = LF_xy_Lane.transpose()
    plt.axis('equal')
    ax[0][1].plot(LF_xy_Lanetp[:,0], LF_xy_Lanetp[:,1])
    ax[0][1].set_title("LF_xy_pretrans")
    
    RF_xy_Lane = np.array([rf_data[lane][1]] + [rf_data[lane][0]])
    RF_xy_Lanetp = RF_xy_Lane.transpose()
    plt.axis('equal')
    ax[0][2].plot(RF_xy_Lanetp[:,0], RF_xy_Lanetp[:,1])
    ax[0][2].set_title("RF_xy_pretrans")

    # dann mit Rotationsmatrix multiplizieren, z.B.
    # xr = 0.99952459*x  - 0.03083182*y
    # yr = 0.03083182*x + 0.99952459*y
    #+ und - allerdings andersrum, weil es sonst in die falsche Richtung
    #rotiert wird
    
    #depending on what the slope of the linear regression is, apply the right rotation matrix
    if slope > 0: # korrigiert die Rotation gegen den Uhrzeigersinn
        COM_xy_Lanetp_new = np.array([(COM_xy_Lanetp[:,0] * rot_d) - (COM_xy_Lanetp[:,1] * rot_c)] + 
                                    [(COM_xy_Lanetp[:,1] * rot_a) + (COM_xy_Lanetp[:,0] * rot_b)])
        plt.axis('equal')
        ax[1][0].plot(COM_xy_Lanetp_new[0,:], COM_xy_Lanetp_new[1,:])
        ax[1][0].set_title("COM_xy_posttrans")

        LF_xy_Lanetp_new = np.array([(LF_xy_Lanetp[:,0] * rot_d) - (LF_xy_Lanetp[:,1] * rot_c)] + 
                                    [(LF_xy_Lanetp[:,1] * rot_a) + (LF_xy_Lanetp[:,0] * rot_b)])
        plt.axis('equal')
        ax[1][1].plot(LF_xy_Lanetp_new[0,:], LF_xy_Lanetp_new[1,:])
        ax[1][1].set_title("LF_xy_posttrans")

        
        RF_xy_Lanetp_new = np.array([(RF_xy_Lanetp[:,0] * rot_d) - (RF_xy_Lanetp[:,1] * rot_c)] + 
                                    [(RF_xy_Lanetp[:,1] * rot_a) + (RF_xy_Lanetp[:,0] * rot_b)])
        plt.axis('equal')
        ax[1][2].plot(RF_xy_Lanetp_new[0,:], RF_xy_Lanetp_new[1,:])
        ax[1][2].set_title("RF_xy_posttrans")
        
        if plot == 'off':
            plt.close()
        else:
            print("---- Lane correction ----")
            plt.show()
        return COM_xy_Lanetp_new, LF_xy_Lanetp_new, RF_xy_Lanetp_new
    
    # ---------------
    elif slope < 0: # korrigiert die Rotation im Uhrzeigersinn
        COM_xy_Lanetp_new = np.array([(COM_xy_Lanetp[:,0] * rot_d) + (COM_xy_Lanetp[:,1] * rot_c)] + 
                                    [(COM_xy_Lanetp[:,1] * rot_a) - (COM_xy_Lanetp[:,0] * rot_b)])
        plt.axis('equal')
        ax[1][0].plot(COM_xy_Lanetp_new[0,:], COM_xy_Lanetp_new[1,:])
        ax[1][0].set_title("COM_xy_posttrans")
        
        
        LF_xy_Lanetp_new = np.array([(LF_xy_Lanetp[:,0] * rot_d) + (LF_xy_Lanetp[:,1] * rot_c)] + 
                                    [(LF_xy_Lanetp[:,1] * rot_a) - (LF_xy_Lanetp[:,0] * rot_b)])
        plt.axis('equal')
        ax[1][1].plot(LF_xy_Lanetp_new[0,:], LF_xy_Lanetp_new[1,:])
        ax[1][1].set_title("LF_xy_posttrans")
        
        
        RF_xy_Lanetp_new = np.array([(RF_xy_Lanetp[:,0] * rot_d) + (RF_xy_Lanetp[:,1] * rot_c)] + 
                                    [(RF_xy_Lanetp[:,1] * rot_a) - (RF_xy_Lanetp[:,0] * rot_b)])
        plt.axis('equal')
        ax[1][2].plot(RF_xy_Lanetp_new[0,:], RF_xy_Lanetp_new[1,:])
        ax[1][2].set_title("RF_xy_posttrans")
        
        if plot == 'off':
            plt.close()
        else:
            print("---- Lane correction ----")
            plt.show()
        return COM_xy_Lanetp_new, LF_xy_Lanetp_new, RF_xy_Lanetp_new
    else:
        print('There is an error. Your line is completely straight.')
        return KeyError

## Calculate Peaks and Distances
def get_peaks(lane, com_data, lf_data, rf_data, plot = 'off'):
    #peaks_COM, _ = find_peaks(com_data[lane][2][1,:], height=0.8, width=10) #!# check height # ??? #may need to change to percentages to detect more peaks
    #peaks_COM, _ = find_peaks(com_data[lane][2][1,:], height=np.mean(com_data[lane][2][1]), width=10) #height = np.mean(COM_z) ###CHANGED
    peaks_COM, _ = find_peaks(com_data[lane][2][1, :], height=0.7)
    
    Peaks2D = com_data[lane][2][:,peaks_COM]
    peaks_timeonly = Peaks2D[0,:]
    peaks_heightonly = Peaks2D[1,:]
    #plt.plot(com_data[lane][2][0,:], com_data[lane][2][1,:])
    #plt.plot(peaks_timeonly, peaks_heightonly, "x")
    #plt.close()
    #plt.show()

    # Left Foot
    peaks_left, _ = find_peaks(lf_data[lane][2], height = 0.10, width=8, distance = 50) #### Height?! 
    #peaks_left, _ = find_peaks(lf_data[lane][2], height = max(lf_data[lane][2]) - 0.04, width=10) #### not good to start from max and look down, some files have high maxes and smaller peak
    ind = peaks_left
    # per-step floor-corrected lift heights (one entry per detected peak)
    peaks_left_corrected = lf_data[lane][2][ind] - np.min(lf_data[lane][2])

    # Right Foot
    peaks_right, _ = find_peaks(rf_data[lane][2], height = 0.10, width=8, distance = 50) #### Height?!
    #peaks_right, _ = find_peaks(rf_data[lane][2], height = max(rf_data[lane][2]) - 0.04, width=10) #### Height?! #height=max(RF_z_Lane1_2)-0.02 ###CHANGED

    ind = peaks_right
    peaks_right_corrected = rf_data[lane][2][ind] - np.min(rf_data[lane][2])

    #find minima 
    RF_z_Lane_transform = rf_data[lane][2]*-1 #weil wir die minima brauchen, die Funktion aber nur peaks finden kann
    minima_RF, _ = find_peaks(RF_z_Lane_transform, height=-0.1, width=30, distance = 40) #!# width & height OLD
    #minima_RF, _ = find_peaks(RF_z_Lane_transform, height=-0.1, width=30, distance = (1/8) * len(RF_z_Lane_transform)) #!# width & height NEW

    LF_z_Lane_transform = lf_data[lane][2]*-1 #weil wir die minima brauchen, die Funktion aber nur peaks finden kann
    minima_LF, _ = find_peaks(LF_z_Lane_transform, height=-0.1, width=30, distance = 40) #!# width & height OLD
    #minima_LF, _ = find_peaks(LF_z_Lane_transform, height=-0.1, width=30, distance = (1/8) * len(LF_z_Lane_transform)) #!# width & height NEW
    
    
    # Plot peaks and minima for control 
    #isn't this all accelerometry data
    if plot == 'show':
        print("---- Peaks and Minima ----")
        fig, (ax1, ax2, ax3)= plt.subplots(3,1)
        ax1.plot(rf_data[lane][2])
        ax1.plot(peaks_right, rf_data[lane][2][peaks_right], 'x')
        ax1.plot(minima_RF, RF_z_Lane_transform[minima_RF]*-1, 'x')
        ax1.set_title('Right Foot')
        
        ax2.plot(lf_data[lane][2])
        ax2.plot(peaks_left, lf_data[lane][2][peaks_left], 'x')
        ax2.plot(minima_LF, LF_z_Lane_transform[minima_LF]*-1, 'x')
        ax2.set_title('Left Foot')
        
        ax3.plot(com_data[lane][2][1])
        ax3.set_title('COM Data')


        plt.tight_layout()
        plt.show()
    
    return peaks_COM, peaks_left, peaks_right, peaks_left_corrected, peaks_right_corrected, peaks_timeonly, peaks_heightonly, minima_RF, minima_LF

def central_diff(s, h = 1):
    return (s[2:] - 2 * s[1:-1] + s[:-2]) / h**2

def intersect(lane, rf_data, lf_data, minima_RF, minima_LF, plot = "off"):
    min_r_vals = rf_data[lane][2][minima_RF]
    min_l_vals = lf_data[lane][2][minima_LF]
    
    '''
    plt.plot(rf_data[lane][2], label = "original signal")
    plt.plot(np.gradient(np.gradient(rf_data[lane][2])), label = "np.gradient twice")
    plt.plot(central_diff(rf_data[lane][2]), label = "central diff")
    plt.legend()
    '''
       
    threshold_l = np.mean(min_l_vals) + 0.02
    threshold_r = np.mean(min_r_vals) + 0.02

    #transform data
    rf_data_trans = np.abs(rf_data[lane][2] - threshold_r) * -1
    lf_data_trans = np.abs(lf_data[lane][2] - threshold_l) * -1

    #peaks of transformed data are closest to threshold
    peaks_th_r, _ = find_peaks(rf_data_trans, height = -0.005)
    peaks_th_l, _ = find_peaks(lf_data_trans, height = -0.005)

    if plot == "show":
        
        #testing why are some nans???
        if np.any(np.isnan(rf_data_trans)) or np.any(np.isnan(lf_data_trans)):
            print("rf_data", minima_RF) #nans????
            print("lf_data", minima_LF)
        
        fig, (ax1, ax2)= plt.subplots(2,1)
        ax1.plot(rf_data_trans)
        ax1.plot(peaks_th_r, rf_data_trans[peaks_th_r], "x")
        ax1.set_title('Right Foot')

        ax2.plot(lf_data_trans)
        ax2.plot(peaks_th_l, lf_data_trans[peaks_th_l], "x")
        ax2.set_title('Left Foot')

        plt.tight_layout()
        plt.show()
    
    return peaks_th_r, peaks_th_l, threshold_r, threshold_l

#also calculates double support time
def calculate_stance_swing(lane, intersect_idx_L, intersect_idx_R, lf_data, rf_data, com_data):
    swing_R = []
    stance_R = []
    swing_L = []
    stance_L = []
    stance_R_idx = []
    stance_L_idx = []

    #assumption is that there's a constant sampling frequency (verify with clara)
    dst_base_R = np.zeros(len(com_data[lane][0][0])) #array of zeros. replace with 1 where right stance is stance
    dst_base_L = np.zeros(len(com_data[lane][0][0])) #... replace zeroes with 1 where left foot is stance
    
    
    for i in range(len(intersect_idx_R)-1):
        #set indicies
        ind = intersect_idx_R[i]
        next = intersect_idx_R[i+1]
        start_t = com_data[lane][0][0][ind]
        end_t = com_data[lane][0][0][next]
        #calc time difference
        dt = end_t - start_t
        if rf_data[lane][2][ind+1] > rf_data[lane][2][ind]: #swing time (if next index in z axis meausurement for right foot is higher)
            swing_R.append(dt)
        else: #stance time
            stance_R.append(dt)
            dst_base_R[ind:next] = 1

    for i in range(len(intersect_idx_L)-1):
        ind = intersect_idx_L[i]
        next = intersect_idx_L[i+1]
        start_t = com_data[lane][0][0][ind]
        end_t = com_data[lane][0][0][next]
        #calc time difference
        dt = end_t - start_t
        if lf_data[lane][2][ind+1] > lf_data[lane][2][ind]: #swing time
            swing_L.append(dt)
        else:
            stance_L.append(dt)
            dst_base_L[ind:next] = 1
    
    dst_base =  dst_base_R + dst_base_L
            
    return swing_R, stance_R, swing_L, stance_L, dst_base

#calculate double support time
def double_support_time(base):
    base_trim = np.trim_zeros(base) #remove end zeros corersponding to only one foot on plate
    dst_times = np.where(base_trim == 2, base_trim, 0)
    dst = (np.sum(dst_times)/2) / len(dst_times)
    return dst

#calculates percent of time spent in swing or stance for each leg
def calculate_percent_cycle(sw_R, st_R, sw_L, st_L):
    sum_R = np.sum(sw_R) + np.sum(st_R)
    sum_L = np.sum(sw_L) + np.sum(st_L)
    sum_all = sum_R + sum_L
    
    ''' probably not needed
    #percent swing and stance R
    percent_sw_R = np.sum(sw_R) / sum_R
    percent_st_R = np.sum(st_R) / sum_R
    
    #percent swing and stance L
    percent_sw_L = np.sum(sw_L) / sum_L
    percent_st_L = np.sum(st_L) / sum_L
    '''
    
    #percent swing and stance overall
    percent_sw = (np.sum(sw_R) + np.sum(sw_L)) / sum_all
    percent_st = (np.sum(st_R) + np.sum(st_L)) / sum_all
    
    return percent_sw, percent_st
    
## Calculates 1/2 of stance time, 
# DEPRECATED ---
def calc_stance_time(lane, rf_data, lf_data, com_data, minima_RF, minima_LF, plot = "show"):
    threshold_r = np.min(rf_data[lane][2]) + 0.02 #accelerometry? manually chosen
    threshold_l = np.min(lf_data[lane][2]) + 0.02 #accelerometry?
    
    #calculate threshold values "crossings" of horizontal
    cross_ind_l = []
    for ind in minima_LF:
        cross_loc = crossing(ind, lf_data[lane][2], threshold_l)
        if cross_loc != None:
            cross_ind_l.append(cross_loc)
    
    cross_ind_r = []
    for ind in minima_RF:
        cross_loc = crossing(ind, rf_data[lane][2], threshold_r)
        if cross_loc != None:
            cross_ind_r.append(cross_loc)
    
    if plot == 'show':
        print("---- Threshold ----")
        fig, (ax1, ax2)= plt.subplots(2,1)
        ax1.plot(rf_data[lane][2])
        ax1.plot(cross_ind_r, rf_data[lane][2][cross_ind_r], 'o')
        ax1.axhline(threshold_r)
        ax1.set_title('Right Foot')
        
        ax2.plot(lf_data[lane][2])
        ax2.plot(cross_ind_l, lf_data[lane][2][cross_ind_l], 'o')
        ax2.axhline(threshold_l)
        ax2.set_title('Left Foot')
        
        plt.tight_layout()
        plt.show()
    
    print("LEFT CROSSINGS", cross_ind_l)
    print("RIGHT CROSSINGS", cross_ind_r)

    stand_phase_times_l = []
    for i in range(len(cross_ind_l)):
        start_t = com_data[lane][0][0][minima_LF[i]]
        end_t = com_data[lane][0][0][cross_ind_l[i]]
        print("LEFT:", start_t, end_t)
        stand_phase_times_l.append(end_t - start_t)
    
    stand_phase_times_r = []
    for i in range(len(cross_ind_r)):
        start_t = com_data[lane][0][0, minima_RF[i]]
        end_t = com_data[lane][0][0, cross_ind_r[i]]
        print("RIGHT:", start_t, end_t)
        stand_phase_times_r.append(end_t - start_t)
        
    return stand_phase_times_l, stand_phase_times_r
    
#find crossing point AFTER minimum, will return None if no threshold cross after the minimum
# DEPRECATED ---
def crossing(min_idx, signal, thresh):
    for i in range(min_idx, len(signal)):
        if signal[i] > thresh:
            return i
    
def calc_step_time_freq(lane, com_data, lf_data, rf_data,):
    # per-step inter-peak intervals (one entry per consecutive pair of COM-z peaks)
    peaks_COM, peaks_left, peaks_right, peaks_left_corrected, peaks_right_corrected, peaks_timeonly, peaks_heightonly, minima_RF, minima_LF = get_peaks(lane, com_data, lf_data, rf_data, plot = 'off')
    return np.diff(peaks_timeonly)

### Concatenate inter-step intervals across all lanes
def calc_interstep_all(lanes, com_data, lf_data, rf_data,):
    InterStepTime_arr = []
    for lane in lanes:
        InterStepTime_arr += list(calc_step_time_freq(lane, com_data, lf_data, rf_data))
    InterStepTime_avg = np.mean(InterStepTime_arr)
    return InterStepTime_avg, InterStepTime_arr

### Calculate Stride Length
def calculate_step_stride_length(lane, com_data, lf_data, rf_data):
    COM_xy_Lanetp_new, LF_xy_Lanetp_new, RF_xy_Lanetp_new = get_gait(lane, com_data, lf_data, rf_data)
    peaks_COM, peaks_left, peaks_right, peaks_left_corrected, peaks_right_corrected, peaks_timeonly, peaks_heightonly, minima_RF, minima_LF = get_peaks(lane, com_data, lf_data, rf_data, plot = 'off')

    # forward-direction strides per foot (peak-to-peak); np.diff returns empty for <2 peaks
    Stride_length_L = np.abs(np.diff(LF_xy_Lanetp_new[1, peaks_left]))
    Stride_length_R = np.abs(np.diff(RF_xy_Lanetp_new[1, peaks_right]))
    Stride_length = list(Stride_length_L) + list(Stride_length_R)

    return Stride_length, Stride_length_L, Stride_length_R #actually stride lengths

def get_step_lengths(minima_LF, minima_RF, LF_xy_Lanetp_new, RF_xy_Lanetp_new, lane):
    
    left_steps = LF_xy_Lanetp_new[1, minima_LF] #getting LF y values
    right_steps = RF_xy_Lanetp_new[1, minima_RF] #getting RF y values
    
    if int(lane[-1]) % 2 == 0: # moving from far end of force plate
        left_steps = -1 * left_steps
        right_steps =  -1 * right_steps #invert so the math below works

    # distance from center used to measure which foot is first 
    center = np.mean(np.concatenate([left_steps, right_steps])) 
    
    step_pos = [] 

    if np.abs(left_steps[0] - center) > np.abs(right_steps[0] - center): #left first, add first two steps
        step_pos.append(left_steps[0])
        step_pos.append(right_steps[0])
        start = 0
    else:
        step_pos.append(right_steps[0])
        step_pos.append(left_steps[0])
        start = 1

    pointer_L = 1
    pointer_R = 1

    #print("LEFT STEPS: ", left_steps)
    #print("RIGHT STEPS: ", right_steps)
    
    try: #catch any IndexOutOfBounds errors, only happen when double peaks are at the end of the signal
        while not (pointer_L >= len(left_steps) and pointer_R >= len(right_steps)): 
            if start % 2 == 0: #adding a L
                if left_steps[pointer_L] > right_steps[pointer_R-1]: #only add left step if its greater than the previous right step
                    step_pos.append(left_steps[pointer_L])
                else: # double peak, move forward and add that one instead
                    pointer_L += 1
                    step_pos.append(left_steps[pointer_L])
                pointer_L += 1 #move pointer forward regardless of whether step is added or not
            else:
                if right_steps[pointer_R] > left_steps[pointer_L-1]: #ditto above
                    step_pos.append(right_steps[pointer_R])
                else:
                    pointer_R += 1
                    step_pos.append(right_steps[pointer_R])
                pointer_R += 1
                
            start += 1
            #print(step_pos)
            #print("pointerL:", pointer_L)
            #print("pointerR:", pointer_R)
    except:
        print("double peak detected at end of signal, moving on...")
    
    #print("COMBINED: ", step_pos)

    step_lengths = []
    for i in range(len(step_pos)-1):
        step_lengths.append(np.abs(step_pos[i+1] - step_pos[i]))
            
    return step_lengths
    
### Calculate Step Width
def calc_step_width(lane, com_data, lf_data, rf_data,):
    COM_xy_Lanetp_new, LF_xy_Lanetp_new, RF_xy_Lanetp_new = get_gait(lane, com_data, lf_data, rf_data)
    peaks_COM, peaks_left, peaks_right, peaks_left_corrected, peaks_right_corrected, peaks_timeonly, peaks_heightonly, minima_RF, minima_LF = get_peaks(lane, com_data, lf_data, rf_data, plot = 'off')
    
    # Calculate RF positions    
    yPos_right = []
    for i in minima_RF:
        yPos_right.append(RF_xy_Lanetp_new[0,i])
    yPos_right2 = np.array(yPos_right) #Array of y-position right foot


    # Calculate RF positions
    yPos_left = []
    for i in minima_LF:
        yPos_left.append(LF_xy_Lanetp_new[0,i])
    yPos_left2 = np.hstack(yPos_left) #Array of y-position left foot
    Step_width = []
    for i in range(min(len(yPos_right2), len(yPos_left2))):
        Step_width.append((yPos_right2[i]) - (yPos_left2[i]))

    Step_width_abs = abs(np.array(Step_width)) #absolute values
    '''
    # Calculate step width
    print("---- Step width ----")   
    plt.plot((yPos_right2), 'x')
    plt.plot(yPos_left2,'x')
    plt.show()
    plt.close()
    '''
    return Step_width_abs

#measure realtive difference in two different arrays (e.g. right and left step lengths)
def asymmetry(arr1, arr2):
    mu1 = np.mean(arr1)
    mu2 = np.mean(arr2)
    diff = np.abs(mu1 - mu2)
    relative = diff/np.mean([mu1, mu2])
    return relative

#measuring coefficient of variation (sample ddof=1, matches statistics.variance)
def variability(arr):
    mu = np.mean(arr)
    sig = np.std(arr, ddof=1)

    return sig/mu

#pass in an array like COM_x_new_time, subject number, and testtype (2 character)
#outputs a dataframe with lane frames
def get_frames(com, subject, test_type):
    SIZE = len(com[1, :])
    lane_idxs = [] # contains all lanes
    lane_count = 0 # count of lanes
        
    for idx in range(SIZE - 1):
        #prevent index out of bounds errors
        sel = com[1, :][idx]
        next = com[1, :][idx+1]
        
        #8/27/25 --> may need to redo to check values between start and end lanes...
        if sel == 9.999 and next != 9.999: #start of a lane
            lane_idxs.append(int(idx))        
        #if sel != 9.999 and next == 9.999: #end of a lane
        if (sel != 9.999 and next == 9.999) and (lane_idxs): #don't add an "end" unless there's at east one sstart 9/4/25
            lane_idxs.append(int(idx))
            lane_count += 1
            
    #takes care of the last element in COM_x_new_time
    if com[1, :][SIZE-1] != 9.999:
        lane_idxs.append(SIZE-1)
        
    #if length of indices list is odd trim the last one off, so there's an even number 
    #fixed 8/27/25 VV
    if len(lane_idxs) % 2 != 0: 
        lane_idxs = lane_idxs[:-1]

    #TODO: just take first 6 lanes
    
    lane_idxs_re = np.reshape(lane_idxs, (-1, 2)).astype(int) #[[start1, end1], [start2,end2]...]
    #print(lane_idxs_re)
    lane_idxs_re_df = pd.DataFrame(lane_idxs_re)
    lane_number = np.array(range(len(lane_idxs_re_df))) + 1
    subj_list = [subject] * len(lane_idxs_re_df)
    test_type_list = [test_type] * len(lane_idxs_re_df)

    df = pd.DataFrame({"subject": subj_list,
                    "test_type": test_type_list,
                    "lane": lane_number})

    patient_frame = pd.concat([df, lane_idxs_re_df], axis = 1)
    patient_frame.columns = ["subject", "test", "lane", "start", "end"]
    
    #returns in dataframe format and flat format
    return [patient_frame, lane_idxs_re] 