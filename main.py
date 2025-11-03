import wfdb
import numpy
import numpy._core.numeric as _nx
import cvxEDA
import pylab as pl
import scipy as sp
from scipy.signal import find_peaks
from scipy.stats import zscore

EDA = [4]

def custom_split(ary:list, number_of_entries:int):
        """
        Splits an array into multiple arrays. 

        Please refer to the ''split'' documentation in the numpy library. The only difference
        between these functions is that this function splits an array into sub arrays with
        a given number of entries. If the last one array would be smaller than the given
        number, it is excluded from the new array.

        returns:
        sub_arys - a numpy array with subarrays of the given length
        """

        Ntotal = len(ary)
        Nsections = int(Ntotal/number_of_entries)
        if Nsections <=0:
                raise ValueError('number selection must be larger then 0.') from None
        section_sizes = number_of_entries
        
        #calculate points, where the array schould be split.
        div_points = []
        for i in range(Ntotal):
            div_points.append(i * section_sizes)

        #do the splitting
        sub_arys = []
        sary = _nx.swapaxes(ary,0,0)
        for i in range(Nsections):
                st = div_points[i]
                try:
                    end = div_points[i+1]
                    sub_arys.append(_nx.swapaxes(sary[st:end], 0, 0))
                except IndexError: #last division point could be out of bound
                    end = Ntotal -1 #do basically nothing

        return sub_arys

def extract_annotation(record:str, extension:str):
        """
        Extract annotation data from an WFDB annotation file.

        returns:
        annotation.sample - the annotation marks as written in the annotation file
        """

        annotation = wfdb.rdann(record, extension)
        return annotation.sample

def components_separation(segment:list, fs=8):
        """
        Separates an EDA signal into its SCL(tonic) and SCR(phasic) components
        as shown in the cvxEDA paper.

        returns: 
        list of returns from the cvxEDA function
        """

        #normalize and separate the components
        segment_norm = zscore(segment)
        [r, p, t, l, d, e, obj] = cvxEDA.cvxEDA(segment_norm, 1./fs)

        return [r, p, t, l, d, e, obj]

def segment_signal(record:str,channel:list, segment_length_s:int):
        """
        Segment a WFDB record of a given channel into given lengths in seconds.
        This function also separates the signal in the SCL,SCR components
        as seen in the cvxEDA module.

        returns:
        stress_signal - a numpy array with the stress signal segments
        non_stress_signal - a numpy array with the non stress signal segments
        """

        #read the signal and extract basic data and annotations
        signal,signal_data = wfdb.rdsamp(record_name=record, channels=channel,warn_empty=True)
        annotation_points = extract_annotation(record=record,extension="atr")
        sampling_frequency = signal_data["fs"]
        signal_length = signal_data["sig_len"]

        #annotation constants for easier readability
        RELAX_ONE_START = annotation_points[0]
        PHYS_START = annotation_points[1]
        RELAX_TWO_START = annotation_points[2]
        COGN_START = annotation_points[3]
        RELAX_THREE_START = annotation_points[5]
        EMOT_START = annotation_points[6]
        RELAX_FOUR_START = annotation_points[7]

        #calculate the length of each segment
        entries_in_segment = int(sampling_frequency * segment_length_s)
        components = components_separation(signal)


        #initialize the arrays for the segments
        non_stress_signal =[]
        stress_signal =[]
        phasic_stress =[]
        phasic_non_stress =[]
        tonic_stress = []
        tonic_non_stress =[]


        #split the data using the custom splitting algorithm
        non_stress_signal.extend(custom_split( signal[RELAX_ONE_START : PHYS_START],entries_in_segment))
        non_stress_signal.extend(custom_split( signal[RELAX_TWO_START : COGN_START],entries_in_segment))
        stress_signal.extend(custom_split( signal[COGN_START : RELAX_THREE_START], entries_in_segment))
        non_stress_signal.extend(custom_split( signal[RELAX_THREE_START : EMOT_START], entries_in_segment))
        stress_signal.extend(custom_split( signal[EMOT_START : RELAX_FOUR_START], entries_in_segment))
        non_stress_signal.extend(custom_split( signal[RELAX_FOUR_START : ], entries_in_segment))

        phasic_non_stress.extend(custom_split( components[0][RELAX_ONE_START : PHYS_START],entries_in_segment))
        phasic_non_stress.extend(custom_split( components[0][RELAX_TWO_START : COGN_START],entries_in_segment))
        phasic_stress.extend(custom_split( components[0][COGN_START : RELAX_THREE_START], entries_in_segment))
        phasic_non_stress.extend(custom_split( components[0][RELAX_THREE_START : EMOT_START], entries_in_segment))
        phasic_stress.extend(custom_split( components[0][EMOT_START : RELAX_FOUR_START], entries_in_segment))
        phasic_non_stress.extend(custom_split( components[0][RELAX_FOUR_START : ], entries_in_segment))

        tonic_non_stress.extend(custom_split( components[2][RELAX_ONE_START : PHYS_START],entries_in_segment))
        tonic_non_stress.extend(custom_split( components[2][RELAX_TWO_START : COGN_START],entries_in_segment))
        tonic_stress.extend(custom_split( components[2][COGN_START : RELAX_THREE_START], entries_in_segment))
        tonic_non_stress.extend(custom_split( components[2][RELAX_THREE_START : EMOT_START], entries_in_segment))
        tonic_stress.extend(custom_split( components[2][EMOT_START : RELAX_FOUR_START], entries_in_segment))
        tonic_non_stress.extend(custom_split( components[2][RELAX_FOUR_START : ], entries_in_segment))       

        return non_stress_signal, stress_signal, phasic_non_stress, phasic_stress, tonic_non_stress, tonic_stress

def group_one_data(directory:str, channel:list, subject_number:int, segment_length:int):
        """
        Groups one subjects data into stress and non stress categories.
        Mostly a wrapper for the segment_signal function.

        returns:
        stress - a numpy array with segments(arrays) in the stress category
        non_stress - a numpy array with segments(arrays) in the non stress category
        """

        #figure out the name of the subject file and extract the data
        record_name = directory + "/Subject" + str(subject_number) +"_AccTempEDA"
        non_stress, stress, phasic_non_stress, phasic_stress, tonic_non_stress, tonic_stress = segment_signal(record=record_name, channel=channel, 
                                        segment_length_s=segment_length)
        return non_stress, stress, phasic_non_stress, phasic_stress, tonic_non_stress, tonic_stress

def group_all_data_by_segments(directory:str, channel:list, data_count:int, segment_length:int):
        """
        Groups all subject data into stress and non stress segments. 
        The data is grouped into these two categories, therefore it cannot be traced
        to a certain subject. 
        Propably mostly for intern use, as group_all_data_by_subject function has in 
        theory more functionality.

        Defines two global arrays stress_segments and non_stress_segments, through which 
        data can be accessed. 

        returns:
        None
        """

        #declare needed arrays
        global stress_segments
        global non_stress_segments
        stress_segments = []
        non_stress_segments = []

        #go through the subject data and group it into the stress and non stress categories
        for i in range(data_count):
                temp_stress, temp_non_stress, _ ,_ = group_one_data(directory, channel, i+1, segment_length)
                non_stress_segments.extend(temp_non_stress)
                stress_segments.extend(temp_stress)

def group_all_data_by_subject(directory:str, channel:list, data_count:int, segment_length:int):
        """
        Groups all data by subject into stress and non stress segments

        Defines a global array subject_data, through which the subject data can be accessed 
        by subject_data[subject_number][stress], in which stress is 1 and non stress 0.

        returns:
        None
        """

        #declare needed arrays
        global subject_data
        subject_data = []
        temp_subject = []

        #go through the subject data and group it into stress and non stress by subject
        for i in range(data_count):
               temp_subject = group_one_data(directory, channel, i+1, segment_length)
               subject_data.append(temp_subject)

def get_subject_data(subject_number:int,type:int, segment_number=500):
        """
        A Function to facilitate getting subject data.

        If the function encounters an IndexError, the whole stress/non stress 
        category of the given subject. 
        The subject_number is expected to be given from 1,...,n, e.g. to 
        access the first subject you have to use 1 as the subject_number
        (not 0, as you would normally as a programmer)

        type=0: non stress segment
        type=1: stress segment
        type=2: phasic non stress segment
        type=3: phasic stress segment
        type=4: tonic non stress segment
        type=5: tonic stress segment


        returns:
        specific segment or subject data
        """

        try:
                return subject_data[subject_number - 1][type][segment_number]
        except IndexError:
                return subject_data[subject_number - 1][type]


def scr_peaks(phasic_segment:list, plot=False):
        """
        Finds all local extrema( peaks and throughs) within an array.
        This function can also plot the given segment with the found extrema.

        returns:
        peaks - list of indices of found peaks
        troughs - list of indices of found troughs
        """

        #find peaks and troughs
        peaks, _ = find_peaks(phasic_segment)
        troughs, _ =find_peaks(-phasic_segment)

        #plot if needed
        if(plot):
                tm = pl.arange(1., len(phasic_segment)+1.)/8
                pl.plot(tm,phasic_segment)
                pl.plot(tm[peaks],phasic_segment[peaks],"x")
                pl.plot(tm[troughs],phasic_segment[troughs],"o")
                pl.show()

        return peaks, troughs

def calculate_scr_features(phasic_segment:list):
        """
        Calculate additional SCR features to form the feature vector.
        For definitions of these features see:
        STRESS DETECTION THROUGH WRIST-BASED ELECTRODERMAL ACTIVITY MONITORING AND MACHINE LEARNING

        returns:
        scr_amplitudes - list of all measured amplitudes in the segment
        scr_onsets - list of all measured onsets in the segment
        scr_recoveries - list of all measured recovery times in s
        """
        #not finished
        peaks, throughs = scr_peaks(phasic_segment,True)
        scr_amplitudes = []
        scr_onsets = []
        scr_recoveries = []
        
        return scr_amplitudes, scr_onsets, scr_recoveries
                


def form_feature_vector(segment:list, phasic_segment:list):
        """
        Forms the Feature vector as seen in the paper
        STRESS DETECTION THROUGH WRIST-BASED ELECTRODERMAL ACTIVITY MONITORING AND MACHINE LEARNING

        returns:
        feature vector as seen in the paper above
        """
        scr_amplitudes, scr_onsets,scr_recoveries = calculate_scr_features(phasic_segment)
        
        return [segment.mean(), segment.min(), segment.max(), segment.std(), scr_amplitudes.mean(), scr_onsets.mean(), scr_recoveries.mean()]

def plot_segment(segment:list, phasic_segment:list, tonic_segment:list):
        """
        Plots a given segment with the phasic components.
        """
        tm = pl.arange(1., len(segment)+1.) / 8
        pl.plot(tm, segment, color='b') #blue
        pl.plot(tm, phasic_segment, color='r') #red
        pl.plot(tm, tonic_segment, color='k') #black
        pl.show()

        return True

def test_cases():
        #test for grouping data by segments
        group_all_data_by_segments(directory="data", channel=EDA, data_count=20, segment_length=30)
        a = len(stress_segments)
        b = len(non_stress_segments)
        print(a)
        print(b)
        print(a+b)

        for i in range(a):
                assert(len(stress_segments[i]) == 240)
        for i in range(b):
                assert(len(non_stress_segments[i]) == 240)
        print("finished grouping by segments")

        #test for grouping data by subject
        group_all_data_by_subject(directory="data", channel=EDA, data_count=20, segment_length=30)
        c = len(subject_data)
        print(c)
        assert(c==20)
        for i in range(20):
                for j in get_subject_data(i + 1,True):
                        assert(len(j) == 240)
                for k in get_subject_data(i + 1,False):
                        assert(len(k) == 240)
        print("finished grouping by subject")

#test_cases()
group_all_data_by_subject(directory="data", channel=EDA, data_count=20, segment_length=30)
plot_segment(get_subject_data(1,1,2),get_subject_data(1,3,2),get_subject_data(1,5,2))
form_feature_vector(get_subject_data(1,1,2),get_subject_data(1,3,2))