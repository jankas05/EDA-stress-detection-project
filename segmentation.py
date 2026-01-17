import wfdb
import numpy
import numpy._core.numeric as _nx
from cvxEDA.src.cvxEDA import cvxEDA
import pylab as pl
from scipy.signal import find_peaks
from scipy.stats import zscore
import csv
import neurokit2 as nk

EDA = [4]
FS = 8

def custom_split(ary:list, number_of_entries:int):
        """
        Splits an array into multiple arrays. 

        Please refer to the ''split'' documentation in the numpy library. The only difference
        between these functions is that this function splits an array into sub arrays with
        a given number of entries. If the last one array would be smaller than the given
        number, it is excluded from the new array.

        :param list: The list to be splitted.
        :param number_of_entries: The number of entries after which the split occurs.

        :return sub_arys: A numpy array with subarrays of the given length.
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

        :param record: An WFDB annotation data file. Has to be a path if the file is not in the current directory.
        :param extension: The file type of the annotation file. Usually set to "atr".

        :return sample: The annotation marks as written in the annotation file.
        """

        annotation = wfdb.rdann(record, extension)
        return annotation.sample

def components_separation(segment:list, fs=8):
        """
        Separates an EDA signal into its SCL(tonic) and SCR(phasic) components
        as shown in the cvxEDA paper.
        
        :param segment: A numpy array with the EDA signal.
        :param fs: The sampling frequency of the signal. Defaults to 8Hz.

        :return list: Array of returns from the cvxEDA function
        """

        #normalize and separate the components
        segment_norm = zscore(segment)
        [r, p, t, l, d, e, obj] = cvxEDA(segment_norm, 1./fs)

        return [r, p, t, l, d, e, obj]

def segment_signal(record:str,channel:list, segment_length:int):
        """
        Segment a WFDB record of a given channel into given lengths in seconds.
        This function also separates the signal in the SCL,SCR components
        as seen in the cvxEDA module.

        :param record: A WFDB data file with the signal.
        :param channel: The channels which are to be considered. This attempt focuses on EDA, but there are more signals.
        :param segment_length: The length of each segment in seconds.

        :return non_stress: A 2d numpy array with the non stress signal segments.
        :return stress: A 2d numpy array with the stress signal segments.
        :return phasic_non_stress: A 2d numpy array with the phasic component of non stress segments.
        :return phasic_stress: A 2d numpy array with the the phasic component of stress segments.
        :return tonic_non_stress: A 2d numpy array with the tonic component of non stress segments.
        :return tonic_stress: A 2d numpy array with the tonic component of stress segements.
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
        entries_in_segment = int(sampling_frequency * segment_length)
        components = components_separation(signal)


        #initialize the arrays for the segments
        non_stress =[]
        stress =[]
        phasic_stress =[]
        phasic_non_stress =[]
        tonic_stress = []
        tonic_non_stress =[]


        #split the data using the custom splitting algorithm
        non_stress.extend(custom_split( signal[RELAX_ONE_START : PHYS_START],entries_in_segment))
        non_stress.extend(custom_split( signal[RELAX_TWO_START : COGN_START],entries_in_segment))
        stress.extend(custom_split( signal[COGN_START : RELAX_THREE_START], entries_in_segment))
        non_stress.extend(custom_split( signal[RELAX_THREE_START : EMOT_START], entries_in_segment))
        stress.extend(custom_split( signal[EMOT_START : RELAX_FOUR_START], entries_in_segment))
        non_stress.extend(custom_split( signal[RELAX_FOUR_START : ], entries_in_segment))

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

        return non_stress, stress, phasic_non_stress, phasic_stress, tonic_non_stress, tonic_stress

def group_one_data(directory:str, channel:list, subject_number:int, segment_length:int):
        """
        Groups one subjects data into stress and non stress categories.
        Mostly a wrapper for the segment_signal function.

        :param directory: The directory in which the subject data is stored.
        :param channel: The channels which are to be considered. This attempt focuses on EDA, but there are more signals.
        :param subject_number: The number of the subject. In the case of the UTD dataset can be a number between 1 and 20.
        :param segment_length: The length of each segment in seconds.

        :return non_stress: A 2d numpy array with the non stress signal segments.
        :return stress: A 2d numpy array with the stress signal segments.
        :return phasic_non_stress: A 2d numpy array with the phasic component of non stress segments.
        :return phasic_stress: A 2d numpy array with the the phasic component of stress segments.
        :return tonic_non_stress: A 2d numpy array with the tonic component of non stress segments.
        :return tonic_stress: A 2d numpy array with the tonic component of stress segements.
        """

        #figure out the name of the subject file and extract the data
        record_name = directory + "/Subject" + str(subject_number) +"_AccTempEDA"
        non_stress, stress, phasic_non_stress, phasic_stress, tonic_non_stress, tonic_stress = segment_signal(record=record_name, channel=channel, 
                                        segment_length=segment_length)
        return non_stress, stress, phasic_non_stress, phasic_stress, tonic_non_stress, tonic_stress

def group_all_data_by_segments(directory:str, channel:list, data_count:int, segment_length:int):
        """
        Groups all subject data, counting from the first to the data_count subject, into stress and non stress segments. 
        The data is grouped into these two categories, therefore it cannot be traced
        to a certain subject. 
        Defines two global arrays stress_segments and non_stress_segments, through which 
        data can be accessed. 

        Depreciated and not used in the project.

        :param directory: The directory in which the subject data is stored.
        :param channel: The channels which are to be considered. This attempt focuses on EDA, but there are more signals.
        :param data_count: The number of subjects. 
        :param segment_length: The length of each segment in seconds.

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

        :param directory: The directory in which the subject data is stored.
        :param channel: The channels which are to be considered. This attempt focuses on EDA, but there are more signals.
        :param data_count: The number of subjects. 
        :param segment_length: The length of each segment in seconds.

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

def get_subject_data(subject_number:int, type:int, segment_number=2048):
        """
        A Function to facilitate getting subject data.

        If the function encounters an IndexError, the whole stress/non stress normal/phasic/tonic
        category of the given subject is returned. 
        The subject_number is expected to be given from 1,...,n, e.g. to 
        access the first subject you have to use 1 as the subject_number
        (not 0, as you would normally as a programmer)

        type=0 - non stress segment, 
        type=1 - stress segment, 
        type=2 - phasic non stress segment, 
        type=3 - phasic stress segment, 
        type=4 - tonic non stress segment, 
        type=5 - tonic stress segment

        :param subject_number: The number of the subject.
        :param type: The type of the segment, which will be returned. See the function description.
        :param segment_number: The number of the segment which will be reurned. Can be omitted, to 
        return all segments of the specified type of one subject.

        :return subject_data: specific segment or all segments of the specified type of one subject
        """

        try:
                return subject_data[subject_number - 1][type][segment_number-1]
        except IndexError:
                return subject_data[subject_number - 1][type]

def scr_peaks(phasic_segment:list, plot=False):
        """
        Finds all local extrema( peaks and throughs) within an array.
        This function can also plot the given segment with the found extrema.

        :param phasic_segment: An array with the phasic component of a segment.
        :param plot: A bool, whether to plot the segment or not. Mostly for debugging purposes.
        Defaults to false.

        returns:
        :return peaks: A list of indices of found peaks.
        :return troughs:  A list of indices of found troughs.
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
        Calculate additional SCR features to form the feature vector. Used for the cvxEDA algorithm
        implementation. 

        For definitions of these features see:
        STRESS DETECTION THROUGH WRIST-BASED ELECTRODERMAL ACTIVITY MONITORING AND MACHINE LEARNING

        :param phasic_segment: An array with the phasic component of a segment.

        :return scr_amplitudes: A list of all measured amplitudes of the peaks in the segment.
        :return scr_onsets: A list of all measured onsets in the segment.
        :return scr_recoveries: A list of all measured recovery times in s.
        """

        peaks, troughs = scr_peaks(phasic_segment)
        scr_amplitudes = []
        scr_onsets = []
        scr_recoveries = []
        current_recovery  = 0 
        recovery_found = False

        #go through all troughs
        for i in range(len(troughs)):                
                if(current_recovery > troughs[i]):
                        continue

                #go through all peaks
                current_recovery = 0
                for j in range(len(peaks)):

                        #ensure non are calculated if there are none

                        if (peaks[j] <= troughs[i]): #peak happened before a trough
                                continue #check others

                        else: 
                                current_peak = peaks[j]
                                current_amplitude = phasic_segment[current_peak] - phasic_segment[troughs[i]]
                                #search for half recovery
                                for k in range(current_peak,len(phasic_segment)):
                                        try:
                                                if (phasic_segment[k] > phasic_segment[current_peak]):
                                                        current_peak = k
                                                if ((phasic_segment[k-1] >= current_amplitude) and (current_amplitude > phasic_segment[k+1])): #here lies the error
                                                        recovery_found = True
                                                        current_recovery = k #full recovery happened
                                                        break
                                                else: 
                                                        continue
                                        except IndexError:
                                                break
                                
                                #calculate all the extracted features
                                if (recovery_found):
                                        current_amplitude = phasic_segment[current_peak] - phasic_segment[troughs[i]]
                                        scr_amplitudes.append(current_amplitude)

                                        current_onset = phasic_segment[troughs[i]]
                                        scr_onsets.append(current_onset)

                                        recovery= (current_recovery - current_peak)/FS
                                        scr_recoveries.append(recovery)

                                        current_amplitude = 0
                                        current_peak = 0
                                        current_onset = 0
                                        recovery = 0
                                        recovery_found = False
                                        break
        
        return scr_onsets, scr_amplitudes, scr_recoveries
                
def form_feature_vector(segment:list, phasic_segment:list):
        """
        Forms the Feature vector as seen in the paper.
        The both parameters should correspond to the same segment.

        STRESS DETECTION THROUGH WRIST-BASED ELECTRODERMAL ACTIVITY MONITORING AND MACHINE LEARNING

        :param segment: A numpy array with a segment of the signal.
        :param phasic_segment: A numpy array with the phasic component of a segment.  

        :return feature_vector: A feature vector as seen in the paper in the description.
        """
        #here comes the neurokit implementation
        scr_onsets, scr_amplitudes, scr_recoveries = calculate_scr_features(phasic_segment)
        
        return [segment.mean(), numpy.min(segment), numpy.max(segment), segment.std(), numpy.mean(scr_onsets), numpy.mean(scr_amplitudes), numpy.mean(scr_recoveries)]

def form_database(directory, channel, data_count, segment_length):
        """
        Forms a dictionary called database with feature vectors and some information about them. 
        database = [{subject, seg_mean, seg_min, seg_max, seg_std, rc_onsets, rc_amp, rc_rec, stress},...]

        :param directory: The directory in which the subject data is stored.
        :param channel: The channels which are to be considered. This attempt focuses on EDA, but there are more signals.
        :param data_count: The number of subjects. 
        :param segment_length: The length of each segment in seconds.

        :return database: A database array with dictionaries as entries. See description above the 
        """
        #segment the data, group it and store it into an global array
        group_all_data_by_subject(directory, channel, data_count, segment_length)

        database = []
        #form the database
        for i in range(len(subject_data)):
                for k in range(2): #0 - non stress, 1 - stress
                        for l in range(len(subject_data[i][k])): #go through each segment

                                #form the feature vector and it to the dictionary
                                V = form_feature_vector(get_subject_data(i + 1, k,l + 1), 
                                    get_subject_data(i + 1, k+2, l + 1)) 
                                subject = {'subject': i + 1, 'seg_mean': V[0], 
                                   'seg_min': V[1], 'seg_max': V[2], 'seg_std': V[3],
                                   'rc_onsets': V[4], 'rc_amp': V[5], 'rc_rec': V[6], 'stress': k}
                                database.append(subject)  
        return database
                             
def export_database(file_name,dictionary):
        """
        Export a dictionary to a csv file. Creates a csvfile and stores it
        in the location of this file.

        :param file_name: The name under which the dictionary will be saved.
        :param dictionary: The dictionarz which will be exported as a csv file.

        returns:
        A csv file with all relevant infomation in the current directory.
        """
        with open(file_name, 'w', newline='') as csvfile:
                fieldnames = [ 'subject', 'seg_mean', 'seg_min', 'seg_max', 
                              'seg_std', 'rc_onsets', 'rc_amp', 'rc_rec', 'stress']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dictionary)
        return True


                
def plot_segment(name:str,segment:list, phasic_segment:list, tonic_segment:list, title:str):
        """
        Plots a given segment with the phasic components and saves it under a name as a svg file.

        :param name: The name under which the svg file will be saved.
        :param segment: A numpy array with a segment of the signal.
        :param phasic_segment: A numpy array with the phasic component of a segment. 
        :param tonic_segment: A numpy array with the tonic component of a segment. 
        :param title: The title of the plot.

        returns:
        A visual representation of the segment and a svg file with the representation.
        """
        tm = pl.arange(1., len(segment)+1.) / 8
        pl.plot(tm, segment, color='k', label="Segment") #black
        pl.plot(tm, phasic_segment, color='r', label="Phasic component") #red
        pl.plot(tm, tonic_segment, color='b', label="Tonic component") #blue

        pl.title(title)
        pl.xlabel("Time (s)")
        pl.ylabel("Amplitude")
        pl.legend(loc="lower right")
        pl.savefig(name, format='svg',dpi=300, bbox_inches='tight')
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
database = form_database(directory="data", channel=EDA, data_count=20, segment_length=30)
#export_database("segments.csv", database)

#plot_segment("results/non_stress_segment.svg", get_subject_data(1,0,4), get_subject_data(1,2,4), get_subject_data(1,4,4),"EDA Signal Decomposition - Nonstress Segment")
#plot_segment("results/stress_segment.svg", get_subject_data(1,1,4), get_subject_data(1,3,4), get_subject_data(1,5,4),"EDA Signal Decomposition - Stress Segment") 

plot_segment("results/non_stress_segment_alt.svg", get_subject_data(1,0,16), get_subject_data(1,2,16), get_subject_data(1,4,16),"EDA Signal Decomposition - Nonstress Segment")
plot_segment("results/stress_segment_alt.svg", get_subject_data(1,1,16), get_subject_data(1,3,16), get_subject_data(1,5,16),"EDA Signal Decomposition - Stress Segment") 